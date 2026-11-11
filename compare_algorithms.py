"""
compare_algorithms.py - KIỂM CHỨNG CHÉO 3 CÁCH TÍNH MA TRẬN S

So sánh trên MỌI file netlist có trong thư mục:
  (A) pairwise      : solve_network()        - nối dần từng cặp cổng (engine.py)
  (B) conn-matrix   : solve_network_global() - ma trận kết nối (solver.py, dùng connect_named_matrix của engine.py)
  (C) scikit-rf     : rf.Circuit             - tham chiếu độc lập

Cách (C) dùng ngay netlist đã parse: mỗi linh kiện được dựng lại bằng
primitive của scikit-rf (DefinedGammaZ0.line / delay_open / delay_short /
capacitor / inductor / Network từ file .s2p), rồi nối theo ĐÚNG topo nút
của netlist bằng rf.Circuit. Vì vậy (C) không dùng bất kỳ mã nào của
engine.py / solver.py -> là tham chiếu thật sự độc lập.

Chạy (5 netlist ví dụ có sẵn): python compare_algorithms.py
Chạy với netlist TUỲ Ý (bất kỳ file .net nào, không cần sửa code):
         python compare_algorithms.py my_circuit.net [file2.net ...]
         -> vẫn chạy đầy đủ (A)/(B)/(C) + vẽ sweep_<ten>.png cho mỗi file
         truyền vào; check_theory() chỉ in đối chiếu lý thuyết cho 5 tên
         ví dụ có sẵn, netlist mới sẽ được BỎ QUA bước này (không lỗi).
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skrf as rf
from skrf.media import DefinedGammaZ0

from components import is_lossless, is_passive, is_reciprocal, to_si
from netlist import GND_NAMES, read_netlist
from solver import solve_network, solve_network_global


def _resolve_circuit():
    """
    Lấy class Circuit của scikit-rf một cách bền vững.

    Lý do: tùy bản cài / tùy cách đóng gói, class Circuit có thể KHÔNG được
    re-export ở namespace cấp cao (`skrf.Circuit`) dù module `skrf.circuit`
    vẫn tồn tại. Vì vậy thử lần lượt:
      1) skrf.Circuit            (đường đi thông thường)
      2) skrf.circuit.Circuit    (import trực tiếp từ submodule)
    Nếu cả hai thất bại -> báo lỗi KÈM THÔNG TIN CHẨN ĐOÁN (đường dẫn gói
    đang được nạp + version) để biết ngay là gói bị che hay cài thiếu.
    """
    circuit_cls = getattr(rf, 'Circuit', None)
    if circuit_cls is not None:
        return circuit_cls
    try:
        from skrf.circuit import Circuit as circuit_cls  # noqa: N813
        return circuit_cls
    except Exception as exc:                                    # pragma: no cover
        raise ImportError(
            "Khong tim thay class Circuit trong scikit-rf dang duoc nap.\n"
            f"  skrf.__file__    = {getattr(rf, '__file__', '?')}\n"
            f"  skrf.__version__ = {getattr(rf, '__version__', '?')}\n"
            "Neu duong dan tren KHONG nam trong site-packages thi co file/thu muc\n"
            "ten 'skrf.py' hoac 'skrf/' trong thu muc lam viec dang che goi that:\n"
            "hay doi ten/xoa no. Neu nam trong site-packages thi goi bi cai thieu:\n"
            "  python -m pip uninstall -y scikit-rf skrf\n"
            "  python -m pip install --no-cache-dir scikit-rf\n"
            "(luu y: 'skrf' va 'scikit-rf' la HAI goi khac nhau tren PyPI - phai\n"
            " go bo 'skrf' neu da cai lan.)"
        ) from exc


Circuit = _resolve_circuit()

TOL = 1e-6          # ngưỡng "khớp" cho phần tử lý tưởng (yêu cầu đề bài)
Z0 = 50.0


# =====================================================================
# 1. THAM CHIẾU ĐỘC LẬP: DỰNG LẠI MẠCH BẰNG scikit-rf Circuit
# =====================================================================

def _media(frequency, theta_deg_at_f, z0_line):
    """
    Môi trường TEM không tán sắc có độ dài điện theta(f) = theta0 * f/f0.
    Dùng gamma = 1j*theta(f) với chiều dài d = 1 m -> gamma*d = 1j*theta(f),
    giống beta_l_from_freq() của components.py.
    """
    return DefinedGammaZ0(frequency=frequency, z0_port=Z0, z0=z0_line,
                          gamma=1j * theta_deg_at_f)


def build_skrf_circuit(net, freqs):
    """
    Dựng lại mạch từ đối tượng netlist bằng scikit-rf.
    Trả về rf.Circuit đã nối đầy đủ (kể cả các cổng ngoài).

    Quy ước nối: mỗi NÚT của netlist thành một 'intersection' của Circuit,
    gồm tất cả các (network, port_index) cắm vào nút đó. Nút đất dùng
    rf.Circuit.Ground. Cách này tương đương với việc netlist.py chèn mối
    nối lý tưởng N cổng, nhưng do scikit-rf tự lo -> độc lập hoàn toàn.
    """
    frequency = rf.Frequency.from_f(np.asarray(freqs, dtype=float), unit='hz')
    f = np.asarray(freqs, dtype=float)

    node_pins = {}          # node -> list[(Network, port_index)]
    gnd_count = 0

    def add(node, ntwk, port):
        node_pins.setdefault(node, []).append((ntwk, port))

    for elem in net.elements:
        t, name, nodes, p = elem['type'], elem['name'], elem['nodes'], elem['params']
        gnd = [n.upper() in GND_NAMES for n in nodes]

        if t in ('RES', 'CAP', 'IND'):
            val = to_si(p[{'RES': 'R', 'CAP': 'C', 'IND': 'L'}[t]])
            med = DefinedGammaZ0(frequency=frequency, z0_port=Z0, z0=Z0)
            if t == 'RES':
                ntwk = med.resistor(val, name=name)
            elif t == 'CAP':
                ntwk = med.capacitor(val, name=name)
            else:
                ntwk = med.inductor(val, name=name)
            # linh kiện nối tiếp 2 cổng; nếu một đầu xuống đất -> nối vào Ground
            for k, node in enumerate(nodes):
                if gnd[k]:
                    gnd_count += 1
                    g = Circuit.Ground(frequency, name=f'gnd_{name}_{k}', z0=Z0)
                    add(f'__GND_{name}_{k}', ntwk, k)
                    add(f'__GND_{name}_{k}', g, 0)
                else:
                    add(node, ntwk, k)
            continue

        if t == 'TLINE':
            theta = np.deg2rad(to_si(p['EL'])) * f / to_si(p['F0'])
            ntwk = _media(frequency, theta, to_si(p['Z0'])).line(1, 'm', name=name)
            add(nodes[0], ntwk, 0)
            add(nodes[1], ntwk, 1)
            continue

        if t == 'STUB':
            theta = np.deg2rad(to_si(p['EL'])) * f / to_si(p['F0'])
            med = _media(frequency, theta, to_si(p['Z0']))
            term = p.get('TERM', 'open').lower()
            ntwk = (med.delay_open(1, 'm', name=name) if term == 'open'
                    else med.delay_short(1, 'm', name=name))
            add(nodes[0], ntwk, 0)
            continue

        if t == 'S2P':
            ts = elem['ts']
            s = np.array([ts.s_interp(fv, mode=p.get('INTERP', 'ri').lower()) for fv in f])
            ntwk = rf.Network(frequency=frequency, s=s, z0=Z0, name=name)
            add(nodes[0], ntwk, 0)
            add(nodes[1], ntwk, 1)
            continue

        if t == 'OPEN':
            med = DefinedGammaZ0(frequency=frequency, z0_port=Z0, z0=Z0)
            add(nodes[0], med.open(name=name), 0)
            continue

        if t == 'SHORT':
            med = DefinedGammaZ0(frequency=frequency, z0_port=Z0, z0=Z0)
            add(nodes[0], med.short(name=name), 0)
            continue

        raise NotImplementedError(f"tham chiếu scikit-rf chưa hỗ trợ {t}")

    # Cổng ngoài -> rf.Circuit.Port, đặt theo thứ tự P1, P2, ...
    port_of_ntwk = {}
    for port_num, node in sorted(net.external_ports.items()):
        port = Circuit.Port(frequency, name=f'port{port_num}', z0=Z0)
        port_of_ntwk[id(port)] = port_num
        add(node, port, 0)

    connections = [pins for pins in node_pins.values()]
    _ = gnd_count
    circuit = Circuit(connections)

    # QUAN TRỌNG: rf.Circuit sắp xếp các cổng ngoài theo THỨ TỰ XUẤT HIỆN trong
    # danh sách 'connections' (xem Circuit.port_indexes), KHÔNG theo tên cổng.
    # Thứ tự nút của ta phụ thuộc thứ tự khai báo linh kiện trong netlist
    # (vd branchline.net cho ra P1,P2,P4,P3) nên phải hoán vị kết quả về đúng
    # thứ tự PORT 1..N đã khai báo trong netlist, nếu không sẽ bị "lệch" giả
    # tạo đúng bằng một phép đổi chỗ cổng.
    appear = [port_of_ntwk[id(ntwk)]
              for pins in connections for (ntwk, _) in pins
              if id(ntwk) in port_of_ntwk]
    perm = np.argsort(appear)        # perm[k] = chỉ số trong S của skrf <- PORT k+1
    return circuit, perm


# =====================================================================
# 2. CHẠY MỘT NETLIST QUA CẢ 3 CÁCH
# =====================================================================

def run_case(netfile, max_points=None, verbose=True):
    """
    Trả về dict kết quả: ma trận S của 3 cách + sai khác + thời gian.
    max_points: giới hạn số điểm tần số (để chạy nhanh); None = dùng hết.

    HOÀN TOÀN TỔNG QUÁT: netfile có thể là BẤT KỲ file .net nào (không chỉ
    5 ví dụ có sẵn trong thư mục), chỉ cần đúng cú pháp netlist.py (FREQ,
    PORT, các phần tử RES/CAP/IND/TLINE/MLIN/CLIN/STUB/OPEN/SHORT/S2P). Đây
    là hàm được gọi lại bởi plot_sweep() bên dưới để tự động vẽ PNG.
    """
    net = read_netlist(netfile)
    freqs = net.freqs
    if max_points is not None and freqs.size > max_points:
        idx = np.linspace(0, freqs.size - 1, max_points).astype(int)
        freqs = freqs[idx]
    nf = freqs.size

    # --- (A) pairwise & (B) connection-matrix ---
    t0 = time.perf_counter()
    S_pair = []
    for fv in freqs:
        blocks, conns, ext = net.build_network_at_freq(fv, z0=Z0)
        S_pair.append(solve_network(blocks, conns, ext))
    t_pair = time.perf_counter() - t0
    S_pair = np.array(S_pair)

    t0 = time.perf_counter()
    S_glob = []
    for fv in freqs:
        blocks, conns, ext = net.build_network_at_freq(fv, z0=Z0)
        S_glob.append(solve_network_global(blocks, conns, ext))
    t_glob = time.perf_counter() - t0
    S_glob = np.array(S_glob)

    # --- (C) scikit-rf ---
    t0 = time.perf_counter()
    circuit, perm = build_skrf_circuit(net, freqs)
    S_skrf = circuit.network.s[:, perm, :][:, :, perm]
    t_skrf = time.perf_counter() - t0

    d_ab = np.abs(S_pair - S_glob).max()
    d_ac = np.abs(S_pair - S_skrf).max()
    d_bc = np.abs(S_glob - S_skrf).max()

    res = dict(name=os.path.basename(netfile), n_ports=S_pair.shape[1], nf=nf,
               freqs=freqs, S_pair=S_pair, S_glob=S_glob, S_skrf=S_skrf,
               d_ab=d_ab, d_ac=d_ac, d_bc=d_bc,
               t_pair=t_pair, t_glob=t_glob, t_skrf=t_skrf)

    if verbose:
        print(f"\n{'=' * 78}")
        print(f"{res['name']}  -  {res['n_ports']} cong, {nf} diem tan so "
              f"({freqs[0] / 1e9:.3g} - {freqs[-1] / 1e9:.3g} GHz)")
        print('=' * 78)
        print(f"  max|A - B|  pairwise vs conn-matrix : {d_ab:.3e}   "
              f"{'KHOP' if d_ab < TOL else 'LECH'}")
        print(f"  max|A - C|  pairwise vs scikit-rf   : {d_ac:.3e}   "
              f"{'KHOP' if d_ac < TOL else 'LECH'}")
        print(f"  max|B - C|  conn-mat vs scikit-rf   : {d_bc:.3e}   "
              f"{'KHOP' if d_bc < TOL else 'LECH'}")
        print(f"  thoi gian: pairwise {t_pair:.3f}s | conn-matrix {t_glob:.3f}s "
              f"| scikit-rf {t_skrf:.3f}s")

        # in ma trận S tại tần số giữa dải để xem trực quan
        k = nf // 2
        print(f"\n  S tai f = {freqs[k] / 1e9:.4g} GHz (cach A - pairwise):")
        for row in np.round(S_pair[k], 5):
            print('   ', '  '.join(f"{v.real:+.5f}{v.imag:+.5f}j" for v in row))
        rec = is_reciprocal(S_pair[k])
        print(f"  reciprocal={rec}  passive={is_passive(S_pair[k])}  "
              f"lossless={is_lossless(S_pair[k])}")

    return res


# =====================================================================
# 3. KIỂM TRA LÝ THUYẾT RIÊNG CHO TỪNG MẠCH
# =====================================================================

def check_theory(res):
    """Đối chiếu với giá trị lý thuyết đã biết của từng mạch (nếu có)."""
    name, S, freqs = res['name'], res['S_pair'], res['freqs']
    k0 = int(np.argmin(np.abs(freqs - 2.4e9)))

    if name == 'wilkinson.net':
        exp = np.array([[0, -0.7071j, -0.7071j],
                        [-0.7071j, 0, 0],
                        [-0.7071j, 0, 0]])
        err = np.abs(S[k0] - exp).max()
        print(f"  [LY THUYET] Wilkinson tai {freqs[k0] / 1e9:.3g} GHz: "
              f"max|S - S_ly_thuyet| = {err:.3e} "
              f"({'KHOP' if err < 1e-4 else 'LECH'})")

    elif name == 'branchline.net':
        s = S[k0]
        # Ma trận lý thuyết chuẩn (Pozar, Microwave Engineering, quadrature hybrid):
        #   S = -1/sqrt(2) * [[0, j, 1, 0],
        #                     [j, 0, 0, 1],
        #                     [1, 0, 0, j],
        #                     [0, 1, j, 0]]
        # Với topo của branchline.net: cổng 1 = vào, cổng 2 = THẲNG (-90 độ),
        # cổng 3 = GHÉP (-180 độ), cổng 4 = CÁCH LY.
        r2 = 1 / np.sqrt(2)
        exp = -r2 * np.array([[0, 1j, 1, 0],
                              [1j, 0, 0, 1],
                              [1, 0, 0, 1j],
                              [0, 1, 1j, 0]], dtype=complex)
        err = np.abs(s - exp).max()
        dphi = np.angle(s[1, 0], deg=True) - np.angle(s[2, 0], deg=True)
        print(f"  [LY THUYET] Branch-line tai {freqs[k0] / 1e9:.3g} GHz: "
              f"max|S - S_ly_thuyet| = {err:.3e} ({'KHOP' if err < 1e-4 else 'LECH'})")
        print(f"              |S21|={abs(s[1, 0]):.4f} (thang) |S31|={abs(s[2, 0]):.4f} "
              f"(ghep) - ly thuyet 0.7071;  |S11|={abs(s[0, 0]):.2e} "
              f"|S41|={abs(s[3, 0]):.2e} (cach ly) - ly thuyet 0")
        print(f"              lech pha S21-S31 = {dphi:.2f} do (ly thuyet +90 "
              f"-> ghep vuong pha)")

    elif name == 'filter_lc.net':
        mag = np.abs(S[:, 1, 0])
        fc = 1e9
        i_pass = np.where(freqs <= 0.9 * fc)[0]
        i_stop = np.where(freqs >= 2.0 * fc)[0]
        ripple_db = -20 * np.log10(mag[i_pass].min())
        att_db = -20 * np.log10(mag[i_stop].max())
        print(f"  [LY THUYET] LPF Chebyshev bac 5: ripple dai thong "
              f"(f<=0.9GHz) = {ripple_db:.3f} dB (thiet ke 0.5 dB), "
              f"suy giam f>=2GHz >= {att_db:.1f} dB")
        # kiểm tra không tổn hao: |S11|^2 + |S21|^2 = 1 (chỉ có L, C)
        uni = np.abs(np.abs(S[:, 0, 0]) ** 2 + np.abs(S[:, 1, 0]) ** 2 - 1).max()
        print(f"              kiem tra khong ton hao |S11|^2+|S21|^2-1: "
              f"max = {uni:.3e}")

    elif name == 'filter_stub.net':
        mag_db = 20 * np.log10(np.abs(S[:, 1, 0]))
        k_f0 = int(np.argmin(np.abs(freqs - 2.4e9)))
        print(f"  [LY THUYET] Loc stub lambda/4: |S21| tai f0=2.4GHz = "
              f"{mag_db[k_f0]:.3f} dB (mong doi ~0 dB, stub -> ho mach), "
              f"tai 1.2GHz = {mag_db[int(np.argmin(np.abs(freqs - 1.2e9)))]:.1f} dB")
        uni = np.abs(np.abs(S[:, 0, 0]) ** 2 + np.abs(S[:, 1, 0]) ** 2 - 1).max()
        print(f"              kiem tra khong ton hao |S11|^2+|S21|^2-1: "
              f"max = {uni:.3e}")

    elif name == 'amplifier.net':
        S21 = S[:, 1, 0]
        S12 = S[:, 0, 1]
        k_f0 = int(np.argmin(np.abs(freqs - 2.4e9)))
        print(f"  [KIEM TRA] Tang khuech dai (KHONG thuan nghich): tai 2.4 GHz "
              f"|S21| = {abs(S21[k_f0]):.4f} ({20 * np.log10(abs(S21[k_f0])):.2f} dB), "
              f"|S12| = {abs(S12[k_f0]):.4f}")
        print(f"              |S21/S12| = {abs(S21[k_f0] / S12[k_f0]):.1f}x  -> "
              f"reciprocal = {is_reciprocal(S[k_f0])} (phai la False)")
        print(f"              max|S21 - S12| tren toan dai = "
              f"{np.abs(S21 - S12).max():.4f} (phai LON, khac 0 ro ret)")


# =====================================================================
# 3b. QUÉT TOÀN DẢI: VẼ ĐỒ THỊ S(f) (biên độ dB + pha) cho CẢ 3 CÁCH
# =====================================================================

# Các mốc đơn vị tần số (Hz), từ lớn đến nhỏ, dùng để tự động chọn đơn vị
# "dễ đọc" (kHz/MHz/GHz/THz) thay cho ký hiệu khoa học (vd 5e8 -> 500 MHz).
_FREQ_UNITS = [(1e12, "THz"), (1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")]


def _fmt_freq(f_hz, sig=3):
    """Đổi 1 giá trị tần số (Hz) thành chuỗi dễ đọc, vd 5e8 -> '500 MHz'."""
    f_hz = abs(float(f_hz))
    if f_hz <= 0:
        return "0 Hz"
    for scale, unit in _FREQ_UNITS:
        if f_hz >= scale:
            return f"{f_hz / scale:.{sig}g} {unit}"
    return f"{f_hz:.{sig}g} Hz"


def _pick_freq_unit(freqs):
    """
    Chọn 1 đơn vị CHUNG (kHz/MHz/GHz/THz) cho CẢ trục tần số, dựa trên giá
    trị tần số lớn nhất trong dải quét - để trục x hiển thị số "dễ đọc"
    (vd 0.5, 1.0, 2.0 GHz) thay vì ký hiệu khoa học như 5e8, 5e11.

    Trả về (scale, ten_don_vi): chia freqs/scale để được giá trị theo
    tên_đơn_vị.
    """
    f_max = float(np.max(np.abs(freqs)))
    for scale, unit in _FREQ_UNITS:
        if f_max >= scale:
            return scale, unit
    return 1.0, "Hz"


def plot_sweep(res, outdir='.'):
    """
    Vẽ biên độ |S_ij| (dB) và pha S_ij (độ) theo tần số, dùng KẾT QUẢ ĐÃ
    CÓ sẵn trong dict res (trả về từ run_case() với max_points=None -> quét
    TOÀN DẢI tần số của netlist, đúng ý Phase TODO của README.md).

    Lưới 2x2:
      - [0,0] Biên độ: pairwise (nét liền) vs conn-matrix (nét đứt ĐEN)
        vs scikit-rf (marker 'o', thưa điểm để đối chiếu bằng mắt).
      - [0,1] Pha S_ij: pairwise (nét liền) vs conn-matrix (nét đứt ĐEN)
        vs scikit-rf (marker 'o', thưa điểm) - cùng quy ước màu/nét như
        panel biên độ.
      - [1,0] Sai số |pairwise - conn-matrix| (log) - lý do 2 đường ở [0,0]
        chồng khít hoàn toàn: sai lệch thực tế chỉ ở mức 1e-6 hoặc nhỏ hơn.
      - [1,1] Sai số |conn-matrix - scikit-rf| (log) - đối chiếu ĐỘC LẬP
        với scikit-rf (tham chiếu bên ngoài, không dùng mã của engine.py).

    Trục tần số dùng 1 đơn vị CHUNG dễ đọc (kHz/MHz/GHz/THz, tự động chọn
    theo _pick_freq_unit) thay vì ký hiệu khoa học kiểu 5e8, 5e11.

    Netlist chỉ có 1 điểm tần số (wilkinson.net, POINTS=1) được BỎ QUA vì
    không có gì để "quét" theo f.

    Lưu: outdir/sweep_<ten_netlist_khong_duoi>.png

    HOÀN TOÀN TỔNG QUÁT: hàm này chỉ dùng dữ liệu đã có trong res (từ
    run_case()), không giả định gì về TÊN hay SỐ CỔNG của netlist -> áp
    dụng được cho BẤT KỲ netlist nào đọc được bởi netlist.read_netlist().
    Xem hướng dẫn chạy ở đầu file (vd: python compare_algorithms.py
    my_circuit.net) để tự động gen PNG cho netlist mới, KHÔNG cần sửa code.
    """
    name = res['name']
    freqs = res['freqs']
    S_pair, S_glob, S_skrf = res['S_pair'], res['S_glob'], res['S_skrf']
    nf, n_ports, _ = S_pair.shape

    if nf < 2:
        print(f"  [BO QUA VE DO THI] {name}: chi co {nf} diem tan so (POINTS=1), "
              f"khong du de quet ve S(f)")
        return None

    scale, unit = _pick_freq_unit(freqs)
    f_plot = freqs / scale
    step = max(nf // 25, 1)          # mật độ marker scikit-rf đối chiếu

    fig, ((ax_mag, ax_ph), (ax_err1, ax_err2)) = plt.subplots(2, 2, figsize=(13, 10))
    cmap = plt.get_cmap('tab10')
    pairs = [(i, j) for i in range(n_ports) for j in range(n_ports)]

    for k, (i, j) in enumerate(pairs):
        c = cmap(k % 10)
        lbl = f"S{i + 1}{j + 1}"

        mag_a = 20 * np.log10(np.maximum(np.abs(S_pair[:, i, j]), 1e-12))
        mag_b = 20 * np.log10(np.maximum(np.abs(S_glob[:, i, j]), 1e-12))
        mag_c = 20 * np.log10(np.maximum(np.abs(S_skrf[::step, i, j]), 1e-12))
        # pairwise: nét liền, màu theo từng S_ij
        ax_mag.plot(f_plot, mag_a, color=c, lw=2.2, alpha=0.9, label=lbl)
        # conn-matrix: nét đứt MÀU ĐEN, nét đậm + dash dài để "nổi" rõ
        # trên nét liền (thay vì dùng cùng màu nhạt nên chồng khít vô hình)
        ax_mag.plot(f_plot, mag_b, color='k', lw=1.1, ls=(0, (7, 4)),
                    alpha=0.8, zorder=5)
        ax_mag.plot(f_plot[::step], mag_c, 'o', color=c, ms=3.5, mfc='none', mew=1.0)

        ph_a = np.unwrap(np.angle(S_pair[:, i, j])) * 180.0 / np.pi
        ph_b = np.unwrap(np.angle(S_glob[:, i, j])) * 180.0 / np.pi
        ph_c = np.unwrap(np.angle(S_skrf[::step, i, j])) * 180.0 / np.pi
        # pairwise: nét liền, màu theo từng S_ij (giống panel biên độ)
        ax_ph.plot(f_plot, ph_a, color=c, lw=2.2, alpha=0.9, label=lbl)
        # conn-matrix: nét đứt MÀU ĐEN (giống panel biên độ)
        ax_ph.plot(f_plot, ph_b, color='k', lw=1.1, ls=(0, (7, 4)),
                   alpha=0.8, zorder=5)
        ax_ph.plot(f_plot[::step], ph_c, 'o', color=c, ms=3.5, mfc='none', mew=1.0)

        # (1) Sai lệch thực tế giữa pairwise và conn-matrix (chính là lý do
        # 2 đường ở biên độ bị chồng khít: sai lệch ở mức ~1e-6 hoặc nhỏ hơn)
        err_ab = np.maximum(np.abs(S_pair[:, i, j] - S_glob[:, i, j]), 1e-16)
        ax_err1.semilogy(f_plot, err_ab, color=c, lw=1.4, label=lbl)

        # (2) Sai lệch giữa conn-matrix và scikit-rf (tham chiếu ĐỘC LẬP,
        # không dùng lại mã nguồn của engine.py/solver.py)
        err_bc = np.maximum(np.abs(S_glob[:, i, j] - S_skrf[:, i, j]), 1e-16)
        ax_err2.semilogy(f_plot, err_bc, color=c, lw=1.4, label=lbl)

    ax_mag.set_xlabel(f"Tần số ({unit})")
    ax_mag.set_ylabel("Biên độ |S_ij| (dB)")
    ax_mag.set_title(
        "Biên độ |S_ij|\n"
        "nét liền = pairwise, nét đứt đen = conn-matrix,\n"
        "'o' = scikit-rf (lấy mẫu thưa)"
    )
    ax_mag.grid(alpha=0.3)
    ax_mag.legend(fontsize=8, ncol=2)

    ax_ph.set_xlabel(f"Tần số ({unit})")
    ax_ph.set_ylabel("Pha S_ij (độ)")
    ax_ph.set_title(
        "Pha S_ij\n"
        "nét liền = pairwise, nét đứt đen = conn-matrix,\n"
        "'o' = scikit-rf (lấy mẫu thưa)"
    )
    ax_ph.grid(alpha=0.3)
    ax_ph.legend(fontsize=8, ncol=2)

    ax_err1.set_xlabel(f"Tần số ({unit})")
    ax_err1.set_ylabel("|S_pairwise − S_conn-matrix|")
    ax_err1.set_title("Sai số: pairwise vs connection-matrix")
    ax_err1.axhline(TOL, color='r', ls=':', lw=1.2, label=f"ngưỡng {TOL:.0e}")
    ax_err1.grid(alpha=0.3, which='both')
    ax_err1.legend(fontsize=7, ncol=2)

    ax_err2.set_xlabel(f"Tần số ({unit})")
    ax_err2.set_ylabel("|S_conn-matrix − S_scikit-rf|")
    ax_err2.set_title(
        "Sai số: connection-matrix vs scikit-rf\n"
        "(đối chiếu với tham chiếu độc lập)"
    )
    ax_err2.axhline(TOL, color='r', ls=':', lw=1.2, label=f"ngưỡng {TOL:.0e}")
    ax_err2.grid(alpha=0.3, which='both')
    ax_err2.legend(fontsize=7, ncol=2)

    fig.suptitle(
        f"{name}  —  quét {nf} điểm tần số "
        f"({_fmt_freq(freqs[0])} – {_fmt_freq(freqs[-1])})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = os.path.join(outdir, f"sweep_{os.path.splitext(name)[0]}.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  [DA VE] {out_path}")
    return out_path


# =====================================================================
# 4. CHỨNG MINH LẠI: BẢN LỖI CŨ CHỈ SAI TRÊN MẠCH KHÔNG THUẬN NGHỊCH
# =====================================================================

def demo_bug_on_netlist(netfile='amplifier.net'):
    """
    Thay connect_ports() đang đã sửa bằng BẢN LỖI CŨ (đảo 2 hệ số) và chạy
    lại cùng netlist -> đo xem sai lệch sinh ra bao nhiêu. Cho thấy:
      - mạch thuận nghịch (wilkinson, filter, branchline): sai lệch = 0
      - mạch có transistor (amplifier): sai lệch LỚN
    """
    import engine
    import solver

    def connect_ports_buggy(S, p, q):
        S = np.asarray(S, dtype=complex)
        N = S.shape[0]
        outside = [k for k in range(N) if k != p and k != q]
        D = (1 - S[p, q]) * (1 - S[q, p]) - S[p, p] * S[q, q]
        S_new = np.zeros((len(outside), len(outside)), dtype=complex)
        for a, k in enumerate(outside):
            for b, m in enumerate(outside):
                S_new[a, b] = S[k, m] + (
                    S[k, p] * S[q, m] * (1 - S[q, p])      # ĐẢO (bug cũ)
                    + S[k, q] * S[p, m] * (1 - S[p, q])    # ĐẢO (bug cũ)
                    + S[k, p] * S[p, m] * S[q, q]
                    + S[k, q] * S[q, m] * S[p, p]
                ) / D
        return S_new

    def connect_named_buggy(S, labels, name_p, name_q):
        p, q = labels.index(name_p), labels.index(name_q)
        S_new = connect_ports_buggy(S, p, q)
        return S_new, [labels[k] for k in range(len(labels)) if k != p and k != q]

    net = read_netlist(netfile)
    freqs = net.freqs[::max(1, net.freqs.size // 21)]

    S_fixed, S_buggy = [], []
    for fv in freqs:
        blocks, conns, ext = net.build_network_at_freq(fv, z0=Z0)
        S_fixed.append(solve_network(blocks, conns, ext))

    orig = solver.connect_named
    solver.connect_named = connect_named_buggy
    try:
        for fv in freqs:
            blocks, conns, ext = net.build_network_at_freq(fv, z0=Z0)
            S_buggy.append(solver.solve_network(blocks, conns, ext))
    finally:
        solver.connect_named = orig
    _ = engine

    S_fixed, S_buggy = np.array(S_fixed), np.array(S_buggy)
    d = np.abs(S_fixed - S_buggy).max()
    return d


# =====================================================================
# 5. MAIN
# =====================================================================

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    # Nếu gọi "python compare_algorithms.py <file1.net> [file2.net ...]" thì
    # dùng DANH SÁCH NETLIST TUỲ Ý do người dùng truyền vào (không giới hạn
    # 5 ví dụ có sẵn bên dưới). Mỗi file sẽ tự động được: kiểm chứng chéo 3
    # cách + vẽ PNG sweep_<ten>.png qua plot_sweep() - không cần sửa code.
    # check_theory() chỉ so sánh lý thuyết cho 5 tên netlist ví dụ (có sẵn),
    # netlist mới sẽ được BỎ QUA bước này (không báo lỗi) nhưng VẪN được vẽ
    # đồ thị và kiểm chứng chéo đầy đủ.
    if len(sys.argv) > 1:
        cases = sys.argv[1:]
    else:
        cases = ['wilkinson.net', 'branchline.net', 'filter_lc.net',
                 'filter_stub.net', 'amplifier.net']

    print("=" * 78)
    print("KIEM CHUNG CHEO: pairwise (engine.py) vs connection-matrix vs scikit-rf")
    print(f"Nguong khop TOL = {TOL:.0e}")
    print("=" * 78)

    results = []
    for c in cases:
        r = run_case(c, max_points=41)
        check_theory(r)
        results.append(r)

    # ---------------- quét toàn dải: vẽ đồ thị S(f) ----------------
    print(f"\n{'=' * 78}")
    print("QUET TOAN DAI TAN SO: VE DO THI S(f) (PNG) cho ca 3 cach")
    print("=" * 78)
    for c in cases:
        r_full = run_case(c, max_points=None, verbose=False)
        plot_sweep(r_full, outdir=here)

    # ---------------- bảng tổng hợp ----------------
    print(f"\n{'=' * 78}")
    print("BANG TONG HOP")
    print("=" * 78)
    hdr = f"{'netlist':16s} {'cong':>4s} {'|A-B|':>11s} {'|A-C|':>11s} {'|B-C|':>11s}  ket luan"
    print(hdr)
    print('-' * 78)
    all_ok = True
    for r in results:
        ok = max(r['d_ab'], r['d_ac'], r['d_bc']) < TOL
        all_ok &= ok
        print(f"{r['name']:16s} {r['n_ports']:4d} {r['d_ab']:11.3e} "
              f"{r['d_ac']:11.3e} {r['d_bc']:11.3e}  "
              f"{'3 CACH KHOP' if ok else 'CO SAI KHAC'}")

    # ---------------- bằng chứng bug cũ ----------------
    print(f"\n{'=' * 78}")
    print("CHUNG MINH: ban LOI CU (dao he so) sai o dau")
    print("=" * 78)
    for c in cases:
        d = demo_bug_on_netlist(c)
        verdict = ("bug AN (mach thuan nghich)" if d < 1e-12
                   else "bug LO RO (mach KHONG thuan nghich)")
        print(f"  {c:16s}  max|S_dasua - S_loicu| = {d:.3e}   -> {verdict}")

    # ---------------- đo thời gian 1001 điểm ----------------
    print(f"\n{'=' * 78}")
    print("THOI GIAN TINH VOI 1001 DIEM TAN SO (yeu cau de bai)")
    print("=" * 78)
    for c in cases:
        net = read_netlist(c)
        if net.freqs.size == 1:
            # Chỉ có 1 điểm tần số (vd wilkinson.net @ 2.4 GHz). Không dùng
            # np.full() vì skrf.Frequency yêu cầu tăng dần NGHIÊM NGẶT, nếu
            # không sẽ phát InvalidFrequencyWarning (1001 giá trị giống nhau).
            # Tăng 1 Hz/điểm (lệch tối đa 1 kHz trên 2.4 GHz) để giữ monotonic
            # tăng dần mà không làm sai lệch kết quả đo thời gian.
            f1001 = net.freqs[0] + np.arange(1001, dtype=float)
        else:
            f1001 = np.linspace(net.freqs[0], net.freqs[-1], 1001)

        t0 = time.perf_counter()
        for fv in f1001:
            blocks, conns, ext = net.build_network_at_freq(fv, z0=Z0)
            solve_network(blocks, conns, ext)
        t_a = time.perf_counter() - t0

        t0 = time.perf_counter()
        for fv in f1001:
            blocks, conns, ext = net.build_network_at_freq(fv, z0=Z0)
            solve_network_global(blocks, conns, ext)
        t_b = time.perf_counter() - t0

        t0 = time.perf_counter()
        circ, _ = build_skrf_circuit(net, f1001)
        _ = circ.network.s
        t_c = time.perf_counter() - t0

        print(f"  {c:16s} pairwise {t_a:6.3f}s | conn-matrix {t_b:6.3f}s "
              f"| scikit-rf {t_c:6.3f}s   ({t_a / 1001 * 1e3:.2f} / "
              f"{t_b / 1001 * 1e3:.2f} / {t_c / 1001 * 1e3:.2f} ms moi diem)")

    print(f"\n{'=' * 78}")
    print("KET LUAN: " + ("TAT CA netlist khop ca 3 cach duoi nguong 1e-6."
                          if all_ok else "CON SAI KHAC - xem bang tren."))
    print("=" * 78)
