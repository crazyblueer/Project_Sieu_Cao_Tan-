"""
Module: netlist.py
Chức năng:
1. Đọc netlist dạng text (.net, .txt).
2. Gọi components.py / touchstone.py để lấy ma trận S của từng linh kiện.
3. Tự động chèn mối nối lý tưởng N cổng cho mọi nút có từ 3 chân trở lên.
4. Xuất (blocks, connections, external_ports) cho bộ giải của Người C.

NGỮ PHÁP (mỗi dòng một lệnh; '#', '!' hoặc ';' bắt đầu chú thích):
    FREQ START=1GHz STOP=3GHz POINTS=1001
    PORT <số> <nút>
    RES   <tên> <n1> <n2> R=<giá trị>          ; n2 = 0/GND -> mắc song song xuống đất
    CAP   <tên> <n1> <n2> C=<giá trị>
    IND   <tên> <n1> <n2> L=<giá trị>
    TLINE <tên> <n1> <n2> Z0=<Zc> EL=<độ> F0=<tần số>
    MLIN  <tên> <n1> <n2> W=<m> L=<m> H=<m> ER=<eps_r>
    CLIN  <tên> <n1> <n2> <n3> <n4> Z0E= Z0O= EL=<độ> F0=<tần số>
    STUB  <tên> <n> Z0=<Zc> EL=<độ> F0=<tần số> TERM=open|short
    OPEN  <tên> <n>
    SHORT <tên> <n>
    S2P   <tên> <n1> <n2> FILE=<đường dẫn> [INTERP=ri|ma]

QUY ƯỚC BÀN GIAO:
    blocks      : [(S_matrix, [nhãn chân, ...]), ...]     nhãn chân dạng '<tên>_p<k>'
    connections : [(nhãn A, nhãn B), ...]                  mỗi cặp là 2 chân nối trực tiếp
    ext_ports   : ['EXT_P1', 'EXT_P2', ...]                 xếp theo số cổng; mỗi nhãn EXT xuất
                                                            hiện đúng 1 lần trong connections,
                                                            chân đi cặp với nó là cổng ngoài
"""

import os
import numpy as np
from components import (
    to_si, resistor_series, capacitor_series, inductor_series,
    tline_general, beta_l_from_freq, microstrip, coupled_line,
    load_1port, z_of, stub, junction_n, open_circuit, short_circuit
)
from touchstone import load_s2p

GND_NAMES = {'0', 'GND'}
# Số nút bắt buộc của mỗi loại linh kiện
N_NODES = {'RES': 2, 'CAP': 2, 'IND': 2, 'TLINE': 2, 'MLIN': 2, 'S2P': 2,
           'CLIN': 4, 'STUB': 1, 'OPEN': 1, 'SHORT': 1}
REQUIRED = {'RES': ['R'], 'CAP': ['C'], 'IND': ['L'], 'TLINE': ['Z0', 'EL', 'F0'],
            'MLIN': ['W', 'L', 'H', 'ER'], 'CLIN': ['Z0E', 'Z0O', 'EL', 'F0'],
            'STUB': ['Z0', 'EL', 'F0'], 'S2P': ['FILE'], 'OPEN': [], 'SHORT': []}


class NetlistError(Exception):
    """Lỗi cú pháp hoặc lỗi topo trong netlist."""


class CircuitNetlist:
    def __init__(self):
        self.freqs = np.array([])     # Mảng tần số quét (Hz)
        self.external_ports = {}      # {port_num: node_name}
        self.elements = []            # Danh sách linh kiện
        self.base_dir = '.'           # Thư mục gốc để tìm file .s2p
        self._s2p_cache = {}          # {đường dẫn: Touchstone2P} - đọc file một lần

    # -----------------------------------------------------------------
    # 1. ĐỌC NETLIST
    # -----------------------------------------------------------------
    def parse_file(self, filename):
        """Đọc và bóc tách file netlist."""
        self.base_dir = os.path.dirname(os.path.abspath(filename))
        with open(filename, 'r', encoding='utf-8') as f:
            return self.parse_text(f.read())

    def parse_text(self, text_content):
        """Đọc netlist từ chuỗi."""
        for line_no, line in enumerate(text_content.splitlines(), 1):
            line_str = line
            for mark in ('#', '!', ';'):
                line_str = line_str.split(mark, 1)[0]
            line_str = line_str.strip()
            if not line_str:
                continue
            try:
                self._parse_line(line_str)
            except NetlistError as e:
                raise NetlistError(f"Dòng {line_no}: '{line.strip()}' -> {e}") from None
            except (ValueError, KeyError, IndexError) as e:
                raise NetlistError(f"Dòng {line_no}: '{line.strip()}' -> {e}") from None
        self._check()
        return self

    def _parse_line(self, line):
        tokens = line.split()
        cmd = tokens[0].upper()

        if cmd == "FREQ":
            params = dict(t.split('=', 1) for t in tokens[1:])
            params = {k.upper(): v for k, v in params.items()}
            f_start, f_stop = to_si(params['START']), to_si(params['STOP'])
            points = int(params.get('POINTS', 1001))
            if not 0 < f_start < f_stop:
                raise NetlistError("cần 0 < START < STOP")
            self.freqs = np.linspace(f_start, f_stop, points)

        elif cmd == "PORT":
            port_num, node = int(tokens[1]), tokens[2]
            if port_num in self.external_ports:
                raise NetlistError(f"cổng {port_num} khai báo hai lần")
            if node.upper() in GND_NAMES:
                raise NetlistError("cổng ngoài không được đặt tại nút đất")
            self.external_ports[port_num] = node

        elif cmd in N_NODES:
            name = tokens[1]
            if any(e['name'] == name for e in self.elements):
                raise NetlistError(f"tên linh kiện '{name}' bị trùng")
            nodes = [t for t in tokens[2:] if '=' not in t]
            params = {k.upper(): v for k, v in (t.split('=', 1) for t in tokens[2:] if '=' in t)}
            if len(nodes) != N_NODES[cmd]:
                raise NetlistError(f"{cmd} cần {N_NODES[cmd]} nút, nhận được {len(nodes)}")
            missing = [k for k in REQUIRED[cmd] if k not in params]
            if missing:
                raise NetlistError(f"{cmd} thiếu tham số {missing}")
            elem = {'type': cmd, 'name': name, 'nodes': nodes, 'params': params}
            if cmd == 'S2P':                      # đọc file .s2p MỘT lần lúc parse
                path = params['FILE']
                if not os.path.isabs(path):
                    path = os.path.join(self.base_dir, path)
                if path not in self._s2p_cache:
                    self._s2p_cache[path] = load_s2p(path)
                elem['ts'] = self._s2p_cache[path]
            self.elements.append(elem)

        else:
            raise NetlistError(f"lệnh không hỗ trợ: {cmd}")

    def _check(self):
        """Kiểm tra sau khi đọc xong: phải có FREQ, cổng, và dải tần nằm trong dữ liệu S2P."""
        if self.freqs.size == 0:
            raise NetlistError("thiếu lệnh FREQ")
        if not self.external_ports:
            raise NetlistError("thiếu lệnh PORT")
        for e in self.elements:
            if e['type'] == 'S2P':
                ts = e['ts']
                if self.freqs[0] < ts.freqs[0] or self.freqs[-1] > ts.freqs[-1]:
                    raise NetlistError(f"{e['name']}: dải quét vượt dải dữ liệu của {ts.filename}")

    # -----------------------------------------------------------------
    # 2. MA TRẬN S CỦA MỘT LINH KIỆN
    # -----------------------------------------------------------------
    def _element_block(self, elem, f_val, z0):
        """Trả về (S, [nút ứng với từng cổng của S])."""
        t, p, nodes = elem['type'], elem['params'], elem['nodes']
        gnd = [n.upper() in GND_NAMES for n in nodes]

        if t in ('RES', 'CAP', 'IND'):
            kind = {'RES': 'R', 'CAP': 'C', 'IND': 'L'}[t]
            val = to_si(p[kind])
            if gnd[0] and gnd[1]:
                raise NetlistError(f"{elem['name']}: cả hai đầu đều nối đất")
            if gnd[0] or gnd[1]:                  # mắc song song xuống đất -> khối 1 cổng
                node = nodes[1] if gnd[0] else nodes[0]
                return load_1port(z_of(kind, val, f_val), z0), [node]
            if t == 'RES':
                return resistor_series(val, z0), nodes
            if t == 'CAP':
                return capacitor_series(val, f_val, z0), nodes
            return inductor_series(val, f_val, z0), nodes

        if any(gnd):
            raise NetlistError(f"{elem['name']}: {t} không được nối trực tiếp xuống đất "
                               f"(dùng SHORT hoặc STUB TERM=short)")

        if t == 'TLINE':
            theta = beta_l_from_freq(f_val, to_si(p['F0']), to_si(p['EL']))
            return tline_general(theta, to_si(p['Z0']), z0), nodes
        if t == 'MLIN':
            return microstrip(to_si(p['W']), to_si(p['H']), to_si(p['ER']), to_si(p['L']), f_val, z0), nodes
        if t == 'CLIN':
            theta = beta_l_from_freq(f_val, to_si(p['F0']), to_si(p['EL']))
            return coupled_line(to_si(p['Z0E']), to_si(p['Z0O']), theta, z0), nodes
        if t == 'STUB':
            theta = beta_l_from_freq(f_val, to_si(p['F0']), to_si(p['EL']))
            return stub(theta, to_si(p['Z0']), z0, p.get('TERM', 'open').lower()), nodes
        if t == 'OPEN':
            return open_circuit(), nodes
        if t == 'SHORT':
            return short_circuit(), nodes
        if t == 'S2P':
            return elem['ts'].s_interp(f_val, mode=p.get('INTERP', 'ri').lower()), nodes
        raise NetlistError(f"loại linh kiện không hỗ trợ: {t}")

    # -----------------------------------------------------------------
    # 3. DỰNG MẠNG TẠI MỘT TẦN SỐ (TOPOLOGY RESOLVER)
    # -----------------------------------------------------------------
    def build_network_at_freq(self, f_val, z0=50.0):
        """
        Tại tần số f_val, trả về (blocks, connections, ext_port_labels) cho Người C.
        - Nút có 2 chân   -> nối trực tiếp.
        - Nút có >= 3 chân -> chèn mối nối lý tưởng N cổng (junction_n).
        - Nút có 1 chân   -> báo lỗi (dây treo).
        """
        blocks, connections = [], []
        node_pins = {}

        def add_pin(node, label):
            node_pins.setdefault(node, []).append(label)

        # 3.1 Ma trận S của từng linh kiện
        for elem in self.elements:
            S, port_nodes = self._element_block(elem, f_val, z0)
            pins = [f"{elem['name']}_p{k + 1}" for k in range(S.shape[0])]
            blocks.append((S, pins))
            for node, pin in zip(port_nodes, pins):
                add_pin(node, pin)

        # 3.2 Cổng ngoài, xếp theo số thứ tự
        ext_port_labels = []
        for port_num, node in sorted(self.external_ports.items()):
            label = f"EXT_P{port_num}"
            ext_port_labels.append(label)
            if node not in node_pins:
                raise NetlistError(f"cổng {port_num} đặt tại nút '{node}' không có linh kiện nào")
            add_pin(node, label)

        # 3.3 Duyệt từng nút
        junc_count = 0
        for node, pins in node_pins.items():
            n = len(pins)
            if n == 1:
                raise NetlistError(f"nút '{node}' chỉ có một chân ({pins[0]}) - dây treo")
            if n == 2:
                if pins[0].startswith('EXT_') and pins[1].startswith('EXT_'):
                    raise NetlistError(f"nút '{node}' chỉ nối hai cổng ngoài với nhau")
                connections.append((pins[0], pins[1]))
            else:
                junc_count += 1
                j_name = f"J_{junc_count}"
                j_pins = [f"{j_name}_{k + 1}" for k in range(n)]
                blocks.append((junction_n(n), j_pins))
                for a, b in zip(pins, j_pins):
                    connections.append((a, b))

        return blocks, connections, ext_port_labels

    def sweep(self, z0=50.0):
        """Tiện ích: lần lượt trả về (f, blocks, connections, ext_ports) cho mọi tần số."""
        for f_val in self.freqs:
            yield (f_val,) + self.build_network_at_freq(f_val, z0)


def read_netlist(filename):
    """Hàm tiện ích đọc file netlist."""
    return CircuitNetlist().parse_file(filename)


# =====================================================================
# BỘ TEST TỰ ĐỘNG
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("KIỂM THỬ MODULE NETLIST.PY (NGƯỜI B)")
    print("=" * 60)

    # 1. Wilkinson: mỗi nút IN, OUT1, OUT2 có 3 chân -> 3 mối nối T
    wilkinson = """
    # Bộ chia Wilkinson 2.4 GHz
    FREQ START=1GHz STOP=4GHz POINTS=1001
    PORT 1 IN
    PORT 2 OUT1
    PORT 3 OUT2
    TLINE L1 IN OUT1 Z0=70.71 EL=90 F0=2.4GHz
    TLINE L2 IN OUT2 Z0=70.71 EL=90 F0=2.4GHz
    RES   R1 OUT1 OUT2 R=100
    """
    net = CircuitNetlist().parse_text(wilkinson)
    blocks, conns, ext = net.build_network_at_freq(2.4e9)
    juncs = [pins for S, pins in blocks if pins[0].startswith('J_')]
    assert len(net.freqs) == 1001 and len(blocks) == 6 and len(juncs) == 3 and len(conns) == 9
    assert ext == ['EXT_P1', 'EXT_P2', 'EXT_P3']
    print(f"[PASS] 1. Wilkinson: {len(blocks)} khối (3 mối nối T), {len(conns)} cặp nối, 1001 điểm tần số")

    # 2. LPF LC: tụ nối đất -> khối 1 cổng; nút 'a' có cổng + C1 + L2 -> mối nối 3 cổng
    lpf = """
    FREQ START=10MHz STOP=3GHz POINTS=1001
    PORT 1 a
    PORT 2 c
    CAP C1 a 0 C=1.967p
    IND L2 a b L=12.876n
    CAP C3 b 0 C=6.366p
    IND L4 b c L=12.876n
    CAP C5 c GND C=1.967p
    """
    net = CircuitNetlist().parse_text(lpf)
    blocks, conns, ext = net.build_network_at_freq(1e9)
    one_port = [pins for S, pins in blocks if S.shape == (1, 1)]
    assert len(one_port) == 3, "Tụ nối đất phải thành khối 1 cổng"
    print(f"[PASS] 2. LPF LC: 3 tụ nối đất -> 3 khối 1 cổng, {len(blocks)} khối, {len(conns)} cặp nối")

    # 3. Nút 4 chân -> mối nối 4 cổng; đường ghép 4 nút; stub
    net = CircuitNetlist().parse_text("""
    FREQ START=1GHz STOP=3GHz POINTS=11
    PORT 1 n1
    PORT 2 n2
    PORT 3 n3
    PORT 4 n4
    CLIN CPL n1 n2 n3 n4 Z0E=69.37 Z0O=36.04 EL=90 F0=2GHz
    STUB S1 n1 Z0=50 EL=45 F0=2GHz TERM=open
    STUB S2 n1 Z0=50 EL=30 F0=2GHz TERM=short
    """)
    blocks, conns, ext = net.build_network_at_freq(2e9)
    sizes = sorted(S.shape[0] for S, pins in blocks if pins[0].startswith('J_'))
    assert sizes == [4], "Nút n1 có 4 chân phải sinh mối nối 4 cổng"
    print("[PASS] 3. Đường ghép 4 cổng + 2 stub tại n1 -> mối nối 4 cổng")

    # 4. Phát hiện lỗi
    bad = {
        "dây treo": "FREQ START=1G STOP=2G POINTS=3\nPORT 1 a\nPORT 2 b\nRES R1 a b R=50\nRES R2 b x R=10",
        "thiếu tham số": "FREQ START=1G STOP=2G POINTS=3\nPORT 1 a\nTLINE T1 a b Z0=50 EL=90",
        "sai số nút": "FREQ START=1G STOP=2G POINTS=3\nPORT 1 a\nCLIN C a b Z0E=60 Z0O=40 EL=90 F0=1G",
        "cổng không nối": "FREQ START=1G STOP=2G POINTS=3\nPORT 1 a\nPORT 2 z\nRES R1 a b R=50\nOPEN O1 b",
        "TLINE nối đất": "FREQ START=1G STOP=2G POINTS=3\nPORT 1 a\nTLINE T1 a 0 Z0=50 EL=90 F0=1G",
    }
    for what, txt in bad.items():
        try:
            CircuitNetlist().parse_text(txt).build_network_at_freq(1.5e9)
            raise AssertionError(f"Không phát hiện lỗi: {what}")
        except NetlistError:
            pass
    print(f"[PASS] 4. Phát hiện đúng {len(bad)} loại lỗi netlist")

    print("\n=> TẤT CẢ KIỂM THỬ ĐẠT.")
