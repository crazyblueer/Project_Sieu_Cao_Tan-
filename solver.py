import os
import numpy as np
from components import is_lossless, is_reciprocal, is_passive, renormalize_general
from engine import connect, block_diag, METHODS


def _finish(S, ordered, pin_z0, ext_z0, z0):
    """
    Bước cuối: đưa S (các cổng ngoài có tham chiếu của CHÂN chưa nối, mặc định z0) về
    tham chiếu ext_z0 của từng cổng ngoài (sóng công suất, Kurokawa -
    components.renormalize_general). ext_z0 = None, hoặc trùng sẵn -> trả S nguyên vẹn.
    """
    if ext_z0 is None:
        return S
    z_from = np.array([(pin_z0[p] if pin_z0 is not None else z0) for p in ordered], dtype=complex)
    z_to = np.asarray(ext_z0, dtype=complex)
    if z_to.shape != z_from.shape:
        raise ValueError(f"ext_z0 co {z_to.size} gia tri nhung co {z_from.size} cong ngoai")
    if np.array_equal(z_from, z_to):
        return S
    return renormalize_general(S, z_from, z_to)


def solve_network(blocks, connections, ext_port_labels, pin_z0=None, ext_z0=None, z0=50.0,
                  method="matrix"):
    """
    Giải mạng: gộp khối, nối các cặp chân, giữ lại các cổng ngoài theo thứ tự P1, P2, ...

    blocks, connections, ext_port_labels : từ netlist.build_network() (3 phần tử đầu).
    pin_z0 : dict {nhãn chân: trở kháng tham chiếu (thực/phức)} của các chân khối
             (phần tử thứ 4 của build_network). Chân không có trong dict, hoặc pin_z0 = None: z0.
    ext_z0 : list trở kháng tham chiếu mong muốn của các cổng ngoài, theo thứ tự
             ext_port_labels (phần tử thứ 5 của build_network). S trả về được chuẩn hóa về ext_z0
             (sóng công suất). None: S giữ tham chiếu của các chân nối ra cổng.
    z0     : tham chiếu mặc định của các chân (50 ohm).
    method : "matrix" (mặc định, nối hết cùng lúc) hoặc "pairwise" (nối dần từng cặp) - engine.connect.

    Dùng nhanh:  S = solve_network(*net.build_network(f), method="matrix")
    """
    if method not in METHODS:
        raise ValueError(f"method phai la mot trong {METHODS}, nhan duoc {method!r}")
    big, lab = block_diag(blocks)          # nhãn EXT KHÔNG nằm trong đây

    ext_set = set(ext_port_labels)
    internal = []          # cặp nối thật (cả hai đều là pin thật)
    ext_map = {}           # EXT_Pk -> tên pin thật được phơi ra ngoài
    for a, b in connections:
        a_ext, b_ext = a in ext_set, b in ext_set
        if a_ext and not b_ext:
            ext_map[a] = b
        elif b_ext and not a_ext:
            ext_map[b] = a
        elif not a_ext and not b_ext:
            internal.append((a, b))
        else:
            raise ValueError(f"Hai cong ngoai noi truc tiep: {a},{b}")

    z0_pins = {l: (pin_z0.get(l, z0) if pin_z0 is not None else z0) for l in lab}
    S, lab = connect(big, lab, internal, z0_pins, method=method)

    # còn lại = các pin được chỉ định ra ngoài; sắp theo thứ tự port P1,P2,P3...
    ordered = [ext_map[e] for e in ext_port_labels]
    assert sorted(lab) == sorted(ordered), f"Con thua/thieu cong: {lab} vs {ordered}"
    idx = [lab.index(p) for p in ordered]
    return _finish(S[np.ix_(idx, idx)], ordered, z0_pins, ext_z0, z0)


def solve_netlist_at_freq(net, f_val, z0=50.0, method="matrix"):
    """
    Tiện ích một bước: dựng mạng tại f_val rồi giải, TÔN TRỌNG trở kháng tham chiếu
    từng cổng khai báo trong netlist (PORT n nút Z0=...; cổng không khai báo = z0).

        S = solve_netlist_at_freq(net, 2.4e9)                      # ma trận kết nối
        S = solve_netlist_at_freq(net, 2.4e9, method="pairwise")   # nối dần từng cặp

    Trả về S (N x N) ở tham chiếu net.port_z0 / z0, theo thứ tự PORT 1..N.
    """
    return solve_network(*net.build_network(f_val, z0), z0=z0, method=method)


# ============================================================================
# TỰ KIỂM THỬ:  python solver.py
# ============================================================================

def self_test():
    import tempfile
    from components import tline_general
    from netlist import read_netlist, CircuitNetlist
    from touchstone import save_s2p

    results = []

    def check(name, ok, extra=""):
        results.append(bool(ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")

    here = os.path.dirname(os.path.abspath(__file__))
    nets = ["wilkinson", "branchline", "filter_lc", "filter_stub", "amplifier"]
    port_z0 = {1: "75", 2: "50+10j", 3: "30-5j", 4: "100+40j"}       # thực khác nhau, phức, phần ảo âm
    TOL = 1e-9

    def load(name, z=None):
        net = read_netlist(os.path.join(here, f"{name}.net"))
        if z:
            net.set_port_z0({k: v for k, v in z.items() if k in net.external_ports})
        return net

    def pick(net, n=7):
        return net.freqs[np.unique(np.linspace(0, len(net.freqs) - 1, n).astype(int))]

    # ---- 1. Wilkinson, Z0 mặc định: 2 thuật toán khớp nhau và khớp lý thuyết
    net = load("wilkinson")
    nw = net.build_network(2.4e9)
    S_m = solve_network(*nw)
    S_p = solve_network(*nw, method="pairwise")
    expected = np.array([[0, -0.7071j, -0.7071j], [-0.7071j, 0, 0], [-0.7071j, 0, 0]])
    check("Wilkinson: khop ly thuyet (S21=S31=-0.707j, S11=S23=0)", np.allclose(S_m, expected, atol=1e-3))
    check("Wilkinson: pairwise = connection-matrix", np.abs(S_m - S_p).max() < 1e-13,
          f"(sai khac {np.abs(S_m - S_p).max():.1e})")
    check("Wilkinson: reciprocal/passive/lossless = True/True/False",
          (is_reciprocal(S_m), is_passive(S_m), is_lossless(S_m)) == (True, True, False))
    check("method sai -> ValueError", _raises(ValueError, lambda: solve_network(*nw, method="abc")))

    # ---- 2. 5 netlist, Z0 cổng ngoài PHỨC / KHÁC NHAU
    rng = np.random.default_rng(11)
    for name in nets:
        net0, net = load(name), load(name, port_z0)
        zext = [net.port_z0[k] for k in sorted(net.external_ports)]
        wa = wb = wc = 0.0
        for f in pick(net):
            truth = renormalize_general(solve_netlist_at_freq(net0, f), 50.0, zext)   # 50 ohm rồi chuẩn hóa
            sp = solve_netlist_at_freq(net, f, method="pairwise")
            sm = solve_netlist_at_freq(net, f, method="matrix")
            wa = max(wa, np.abs(sp - truth).max(), np.abs(sm - truth).max())
            # thêm Z0 phức ngẫu nhiên ở MỖI chân khối (khối được chuẩn hóa lại => vật lý không đổi)
            blocks, conns, ext, pin_z0, ext_z0 = net.build_network(f)
            bl2, pz2 = [], {}
            for S, pins in blocks:
                zz = [complex(rng.uniform(20, 150), rng.uniform(-60, 60)) for _ in pins]
                bl2.append((renormalize_general(S, 50.0, zz), pins))
                pz2.update(dict(zip(pins, zz)))
            for m in ("pairwise", "matrix"):
                wb = max(wb, np.abs(solve_network(bl2, conns, ext, pz2, ext_z0, method=m) - truth).max())
            wc = max(wc, np.abs(sp - sm).max())
        check(f"{name}.net: PORT Z0 phuc/khac nhau = dap an vat ly (2 thuat toan)", wa < TOL,
              f"({len(zext)} cong, sai khac {wa:.1e})")
        check(f"{name}.net: + Z0 phuc ngau nhien o moi chan khoi", wb < TOL, f"(sai khac {wb:.1e})")
        check(f"{name}.net: pairwise = connection-matrix", wc < TOL, f"(sai khac {wc:.1e})")

    # ---- 3. khai Z0 tường minh / trong netlist / qua set_port_z0
    n50, n0 = load("branchline", {1: 50, 2: 50, 3: 50, 4: 50}), load("branchline")
    check("PORT Z0=50 tuong minh == khong khai bao (bit-exact)",
          np.array_equal(solve_netlist_at_freq(n50, 2.4e9), solve_netlist_at_freq(n0, 2.4e9)))
    txt = open(os.path.join(here, "filter_lc.net"), encoding="utf-8").read()
    txt = txt.replace("PORT 1 a", "PORT 1 a Z0=75").replace("PORT 2 c", "PORT 2 c Z0=50+10j")
    a = solve_netlist_at_freq(_parse(txt, here), 1e9)
    b = solve_netlist_at_freq(load("filter_lc", {1: 75, 2: "50+10j"}), 1e9)
    check("Z0 khai trong .net == set_port_z0 (bit-exact)", np.array_equal(a, b))
    # khối dựng ở 75 ohm thực, cổng ra 75: đúng
    b75, c75, e75, _, _ = n0.build_network(2.4e9)
    S75 = [(renormalize_general(S, 50.0, 75.0), p) for S, p in b75]
    pz75 = {p: 75.0 for _, ps in S75 for p in ps}
    ref = renormalize_general(solve_netlist_at_freq(n0, 2.4e9), 50.0, 75.0)
    ok = all(np.abs(solve_network(S75, c75, e75, pz75, [75.0] * 4, 75.0, method=m) - ref).max() < TOL
             for m in ("matrix", "pairwise"))
    check("khoi dung o 75 ohm thuc, cong ra 75: dung (2 thuat toan)", ok)
    check("ext_z0 sai so cong -> ValueError",
          _raises(ValueError, lambda: solve_network(b75, c75, e75, None, [50.0, 50.0], 50.0)))

    # ---- 4. S2P (R = 75, lambda/4) giữa 2 cổng 50 ohm == TLINE Z0=75
    with tempfile.TemporaryDirectory() as tmp:
        q = np.array([[0, -1j], [-1j, 0]])
        path = os.path.join(tmp, "q75.s2p")
        save_s2p(path, np.array([1e9, 3e9]), np.array([q, q]), z0=75.0, fmt="RI", freq_unit="GHz")
        t_s2p = f"FREQ START=2G STOP=2G POINTS=1\nPORT 1 a\nPORT 2 b\nS2P X1 a b FILE={path}\n"
        t_tl = "FREQ START=2G STOP=2G POINTS=1\nPORT 1 a\nPORT 2 b\nTLINE T1 a b Z0=75 EL=90 F0=2G\n"
        res = [[solve_netlist_at_freq(CircuitNetlist().parse_text(t), 2e9, method=m) for m in METHODS]
               for t in (t_s2p, t_tl)]
        check("netlist S2P(R=75) == TLINE 75 ohm (2 thuat toan)",
              all(np.allclose(res[0][k], res[1][k], atol=1e-12) for k in (0, 1)))

    print(f"\n=> {sum(results)}/{len(results)} PASS")
    return all(results)


def _raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    return False


def _parse(text, base_dir):
    from netlist import CircuitNetlist
    net = CircuitNetlist()
    net.base_dir = base_dir
    return net.parse_text(text)


if __name__ == "__main__":
    raise SystemExit(0 if self_test() else 1)
