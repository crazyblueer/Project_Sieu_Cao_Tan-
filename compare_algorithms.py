"""
compare_algorithms.py - Double-check 3 cách tính ma trận S:

  (A) pairwise      : solve_network()        - nối dần từng cặp cổng (engine.py)
  (B) conn-matrix   : solve_network_global() - ma trận kết nối (solver.py, dùng connect_named_matrix của engine.py)
  (C) scikit-rf     : rf.Circuit             - tham chieu doc lap

Cách (C) dùng ngay netlist đã parse: mỗi linh kiện được dựng lại bằng
primitive của scikit-rf (DefinedGammaZ0.line / delay_open / delay_short /
capacitor / inductor / Network từ file .s2p), rồi nối theo đúng topo nút
của netlist bằng rf.Circuit. Vì vậy (C) không dùng bất kì module nào của
engine.py / solver.py --> là tham chiếu độc lập để so sánh.

Chay:  python compare_algorithms.py
"""

import os
import time

import numpy as np
import skrf as rf
from skrf.media import DefinedGammaZ0

from components import is_lossless, is_passive, is_reciprocal, to_si
from netlist import GND_NAMES, read_netlist
from solver import solve_network, solve_network_global


def _resolve_circuit():
    """
    Lấy class Circuit của scikit-rf.

    Lý do: tùy version, class Circuit có thể không được re-export ở namespace
    cấp cao (`skfr.Circuit`) dù module `skrf.circuit` vẫn tồn tại. Vì vậy thử lần lượt:
      1) skrf.Circuit            (normally)
      2) skrf.circuit.Circuit    (import trực tiếp từ submodule)
    Nếu cả hai đều failed -> raise error.
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

from components import is_lossless, is_passive, is_reciprocal, to_si
from netlist import GND_NAMES, read_netlist
from solver import solve_network, solve_network_global

TOL = 1e-6          # nguong "khop" cho phan tu ly tuong (yeu cau de bai)
Z0 = 50.0


# =====================================================================
# 1. THAM CHIẾU ĐỘC LẬP: DỰNG LẠI MẠCH BẰNG scikit-rf Circuit
# =====================================================================

def _media(frequency, theta_deg_at_f, z0_line):
    """
    Moi truong TEM khong tan sac co do dai dien theta(f) = theta0 * f/f0.
    Dung gamma = 1j*theta(f) voi chieu dai d = 1 m -> gamma*d = 1j*theta(f),
    giong beta_l_from_freq() cua components.py.
    """
    return DefinedGammaZ0(frequency=frequency, z0_port=Z0, z0=z0_line,
                          gamma=1j * theta_deg_at_f)


def build_skrf_circuit(net, freqs):
    """
    Dung lai mach tu doi tuong netlist bang scikit-rf.
    Tra ve rf.Circuit da noi day du (ke ca cac cong ngoai).

    Quy uoc noi: moi NUT cua netlist thanh mot 'intersection' cua Circuit,
    gom tat ca cac (network, port_index) cam vao nut do. Nut dat dung
    rf.Circuit.Ground. Cach nay tuong duong voi viec netlist.py chen moi
    noi ly tuong N cong, nhung do scikit-rf tu lo -> doc lap hoan toan.
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
            # linh kien noi tiep 2 cong; neu mot dau xuong dat -> noi vao Ground
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

        raise NotImplementedError(f"tham chieu scikit-rf chua ho tro {t}")

    # Cong ngoai -> rf.Circuit.Port, dat theo thu tu P1, P2, ...
    port_of_ntwk = {}
    for port_num, node in sorted(net.external_ports.items()):
        port = Circuit.Port(frequency, name=f'port{port_num}', z0=Z0)
        port_of_ntwk[id(port)] = port_num
        add(node, port, 0)

    connections = [pins for pins in node_pins.values()]
    _ = gnd_count
    circuit = Circuit(connections)

    # QUAN TRONG: rf.Circuit sap xep cac cong ngoai theo THU TU XUAT HIEN trong
    # danh sach 'connections' (xem Circuit.port_indexes), KHONG theo ten cong.
    # Thu tu nut cua ta phu thuoc thu tu khai bao linh kien trong netlist
    # (vd branchline.net cho ra P1,P2,P4,P3) nen phai hoan vi ket qua ve dung
    # thu tu PORT 1..N da khai bao trong netlist, neu khong se bi "lech" gia
    # tao dung bang mot phep doi cho cong.
    appear = [port_of_ntwk[id(ntwk)]
              for pins in connections for (ntwk, _) in pins
              if id(ntwk) in port_of_ntwk]
    perm = np.argsort(appear)        # perm[k] = chi so trong S cua skrf <- PORT k+1
    return circuit, perm


# =====================================================================
# 2. CHẠY 1 NETLIST QUA CẢ 3 CÁCH
# =====================================================================

def run_case(netfile, max_points=None, verbose=True):
    """
    Tra ve dict ket qua: ma tran S cua 3 cach + sai khac + thoi gian.
    max_points: gioi han so diem tan so (de chay nhanh); None = dung het.
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

        # in ma tran S tai tan so giua dai de xem truc quan
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
    """Doi chieu voi gia tri ly thuyet da biet cua tung mach (neu co)."""
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
        # Ma tran ly thuyet chuan (Pozar, Microwave Engineering, quadrature hybrid):
        #   S = -1/sqrt(2) * [[0, j, 1, 0],
        #                     [j, 0, 0, 1],
        #                     [1, 0, 0, j],
        #                     [0, 1, j, 0]]
        # Voi topo cua branchline.net: cong 1 = vao, cong 2 = THANG (-90 do),
        # cong 3 = GHEP (-180 do), cong 4 = CACH LY.
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
        # kiem tra khong ton hao: |S11|^2 + |S21|^2 = 1 (chi co L, C)
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
# 4. CHỨNG MINH LẠI: BẢN LỖI CŨ CHỈ SAI TRÊN MẠCH KHÔNG THUẬN NGHỊCH
# =====================================================================

def demo_bug_on_netlist(netfile='amplifier.net'):
    """
    Thay connect_ports() dang da sua bang BAN LOI CU (dao 2 he so) va chay
    lai cung netlist -> do xem sai lech sinh ra bao nhieu. Cho thay:
      - mach thuan nghich (wilkinson, filter, branchline): sai lech = 0
      - mach co transistor (amplifier): sai lech LON
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
                    S[k, p] * S[q, m] * (1 - S[q, p])      # DAO (bug cu)
                    + S[k, q] * S[p, m] * (1 - S[p, q])    # DAO (bug cu)
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

    # ---------------- bang tong hop ----------------
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

    # ---------------- bang chung bug cu ----------------
    print(f"\n{'=' * 78}")
    print("CHUNG MINH: ban LOI CU (dao he so) sai o dau")
    print("=" * 78)
    for c in cases:
        d = demo_bug_on_netlist(c)
        verdict = ("bug AN (mach thuan nghich)" if d < 1e-12
                   else "bug LO RO (mach KHONG thuan nghich)")
        print(f"  {c:16s}  max|S_dasua - S_loicu| = {d:.3e}   -> {verdict}")

    # ---------------- do thoi gian 1001 diem ----------------
    print(f"\n{'=' * 78}")
    print("THOI GIAN TINH VOI 1001 DIEM TAN SO (yeu cau de bai)")
    print("=" * 78)
    for c in cases:
        net = read_netlist(c)
        if net.freqs.size == 1:
            # Chi co 1 diem tan so (vd wilkinson.net @ 2.4 GHz). Khong dung
            # np.full() vi skrf.Frequency yeu cau tang dan NGHIEM NGAT, neu
            # khong se phat InvalidFrequencyWarning (1001 gia tri giong nhau).
            # Tang 1 Hz/diem (lech toi da 1 kHz tren 2.4 GHz) de giu monotonic
            # tang dan ma khong lam sai lech ket qua do thoi gian.
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
