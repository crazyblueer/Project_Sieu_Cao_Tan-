"""
Module: monte_carlo.py
Chức năng: MÔ PHỎNG MONTE-CARLO DUNG SAI LINH KIỆN (phần mở rộng của đề bài -
"phân tích độ nhạy, dung sai Monte-Carlo, tối ưu, nhiễu").

Ý tưởng:
    - Lấy một netlist (.net) đã có trong SCT/ (filter_lc.net, amplifier.net, ...).
    - Với mỗi "trial", nhiễu ngẫu nhiên giá trị R/L/C theo PHÂN BỐ GAUSS
      (chuẩn/normal) quanh giá trị danh định, với dung sai công bố ±tol_frac
      tương đương ±3*sigma (quy ước công nghiệp phổ biến) -> mô hình đúng
      thực tế sản xuất: phần lớn linh kiện tập trung gần giá trị danh định,
      chỉ hiếm khi lệch mạnh ra biên.
    - Giải lại mạng bằng solver.solve_netlist_at_freq() (mặc định method="matrix" -
      thuật toán connection-matrix, đã kiểm chứng khớp pairwise/scikit-rf tới
      1e-6..1e-16, xem compare_algorithms.py) tại (các) tần số cần khảo sát.
    - Thu thập phân bố |S11|,|S21| (dB) trên toàn bộ trial -> thống kê
      mean/std/percentile + tỷ lệ ĐẠT spec (yield) -> dung sai linh kiện ảnh
      hưởng thế nào đến hiệu năng mạch thực tế (không chỉ giá trị lý tưởng).

Trở kháng tham chiếu (Z0) CỔNG NGOÀI có thể PHỨC và KHÁC NHAU từng cổng:
    - khai trong netlist:  PORT 2 OUT Z0=50+10j   (solver.solve_netlist_at_freq tự xử lý), hoặc
    - truyền qua run_monte_carlo(..., port_z0={1: 75, 2: '50+10j'}) mà KHÔNG phải sửa file .net.
    Không khai -> Z0 = 50 (kết quả y hệt bản cũ). Chỉ Z0 cổng ngoài thay đổi, thuật toán
    giải và nhiễu linh kiện giữ nguyên; |S| (dB) tính theo sóng công suất (Kurokawa).

Dùng lại TRỰC TIẾP các module đã có (không viết lại engine):
    netlist.py    -> đọc .net, build_network(), set_port_z0()
    solver.py     -> solve_netlist_at_freq() (connection-matrix, nhanh nhất)
    components.py -> to_si() để đổi giá trị '5.4303p' -> số thực trước khi nhiễu

Chạy trực tiếp: python3 monte_carlo.py          (demo 2000 trial, vẽ PNG)
                python3 monte_carlo.py --test   (tự kiểm thử, vài giây)
"""

import copy
import textwrap
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from netlist import read_netlist, NetlistError
from solver import solve_netlist_at_freq
from components import to_si


# =====================================================================
# 1. NHIỄU NGẪU NHIÊN MỘT PHẦN TỬ NETLIST (PHÂN BỐ GAUSS)
# =====================================================================
def perturb_value(nominal_str, tol_frac, rng, sigma_rule=3.0, clip_sigma=None):
    """
    Nhiễu MỘT giá trị linh kiện (string SI, ví dụ '5.4303p') quanh giá trị
    danh định, theo PHÂN BỐ GAUSS (chuẩn):

        eps ~ N(0, sigma^2)   với   sigma = tol_frac / sigma_rule

    Quy ước công nghiệp phổ biến: dung sai công bố ±X% tương đương ±3*sigma
    (sigma_rule=3.0 mặc định) -> khoảng 99.7% linh kiện nằm trong ±X%,
    phần còn lại (đuôi phân bố) có thể vượt nhẹ ra ngoài nếu không cắt.
    Nếu muốn GIỚI HẠN cứng (không cho vượt quá dung sai công bố dù hiếm),
    truyền clip_sigma=sigma_rule để cắt (clip) eps vào [-tol_frac, +tol_frac].

    Trả về SỐ THỰC (đơn vị cơ bản SI), KHÔNG trả về string, vì
    netlist.py._element_block() sẽ tự gọi to_si() lại -> truyền số thực
    thì to_si() trả về y nguyên (xem components.to_si: isinstance float).
    """
    nominal = to_si(nominal_str)
    sigma = tol_frac / sigma_rule
    eps = rng.normal(0.0, sigma)
    if clip_sigma is not None:
        bound = clip_sigma * sigma
        eps = np.clip(eps, -bound, bound)
    return nominal * (1.0 + eps)


def make_trial_netlist(net, tol_frac, rng, keys=('R', 'L', 'C'), sigma_rule=3.0,
                        clip_sigma=None):
    """
    Trả về BẢN SAO nông của CircuitNetlist với các tham số R/L/C (nếu có)
    của MỖI linh kiện được nhiễu độc lập theo phân bố Gauss. Dùng
    copy.copy() (shallow) cho đối tượng net -> chia sẻ freqs/
    external_ports/base_dir/_s2p_cache (không đổi, không cần deepcopy,
    tiết kiệm thời gian cho hàng nghìn trial), chỉ thay THUỘC TÍNH
    .elements bằng list mới (deepcopy từng dict linh kiện) để không làm
    hỏng giá trị danh định của net gốc.
    """
    net_trial = copy.copy(net)
    new_elements = []
    for elem in net.elements:
        e2 = dict(elem)
        e2['params'] = dict(elem['params'])
        for key in keys:
            if key in e2['params']:
                e2['params'][key] = perturb_value(
                    e2['params'][key], tol_frac, rng,
                    sigma_rule=sigma_rule, clip_sigma=clip_sigma)
        new_elements.append(e2)
    net_trial.elements = new_elements
    return net_trial


def _fmt_z(z):
    z = complex(z)
    return f"{z.real:g}" if z.imag == 0 else f"{z.real:g}{z.imag:+g}j"


# =====================================================================
# 2. CHẠY N TRIAL TẠI (CÁC) TẦN SỐ KHẢO SÁT
# =====================================================================
def run_monte_carlo(netlist_path, freqs_eval, tol_frac=0.05, n_trials=2000,
                     z0=50.0, seed=123, keys=('R', 'L', 'C'), sigma_rule=3.0,
                     clip_sigma=None, port_z0=None):
    """
    port_z0 : Z0 cổng ngoài (thực/phức, khác nhau từng cổng), xem CircuitNetlist.set_port_z0();
              None = dùng đúng khai báo trong netlist (không khai = z0 = 50).

    Dung sai được mô hình theo PHÂN BỐ GAUSS, sigma = tol_frac / sigma_rule.

    Trả về dict:
        'S_nom'   : S danh định tại mỗi tần số khảo sát, shape (nf, N, N)
        'S_trial' : S của tất cả trial, shape (n_trials, nf, N, N)
        'elapsed' : thời gian chạy (s)
        'ext'     : nhãn cổng ngoài (thứ tự P1,P2,...)
        'z0_ports': Z0 từng cổng ngoài (thứ tự PORT 1..N) mà S tham chiếu tới
    """
    net = read_netlist(netlist_path)
    net.set_port_z0(port_z0)
    z0_ports = [complex(net.port_z0.get(k, z0)) for k in sorted(net.external_ports)]
    freqs_eval = np.atleast_1d(np.asarray(freqs_eval, dtype=float))
    rng = np.random.default_rng(seed)

    # --- danh định ---
    S_nom = []
    ext_ref = None
    for fv in freqs_eval:
        S_nom.append(solve_netlist_at_freq(net, fv, z0, method="matrix"))
        ext_ref = [f"EXT_P{k}" for k in sorted(net.external_ports)]
    S_nom = np.array(S_nom)

    # --- các trial ---
    S_trial = np.empty((n_trials, len(freqs_eval)) + S_nom.shape[1:], dtype=complex)
    t0 = time.perf_counter()
    for i in range(n_trials):
        net_t = make_trial_netlist(net, tol_frac, rng, keys=keys,
                                    sigma_rule=sigma_rule, clip_sigma=clip_sigma)
        for k, fv in enumerate(freqs_eval):
            S_trial[i, k] = solve_netlist_at_freq(net_t, fv, z0, method="matrix")
    elapsed = time.perf_counter() - t0

    return {'S_nom': S_nom, 'S_trial': S_trial, 'elapsed': elapsed,
            'ext': ext_ref, 'freqs_eval': freqs_eval, 'n_trials': n_trials,
            'tol_frac': tol_frac, 'z0_ports': z0_ports}


# =====================================================================
# 3. THỐNG KÊ + VẼ HISTOGRAM
# =====================================================================
def _db(x):
    return 20.0 * np.log10(np.abs(x) + 1e-300)


def _spec_band(limit, nom_db):
    """(lo, hi) dB từ giới hạn spec: ('nom', tol) -> danh định ± tol; (lo, hi) -> giữ nguyên."""
    if isinstance(limit[0], str):
        if limit[0] != 'nom':
            raise ValueError(f"spec: chỉ hỗ trợ 'nom' hoặc (lo, hi), nhận được {limit!r}")
        return nom_db - float(limit[1]), nom_db + float(limit[1])
    return float(limit[0]), float(limit[1])


def _fit_suptitle(fig, title, margin=0.02):
    """
    Đặt tiêu đề chung cho hình; nếu rộng hơn hình thì tự xuống dòng (textwrap) cho đến khi vừa,
    rồi chừa đúng chiều cao của tiêu đề phía trên các ô đồ thị.
    """
    t = fig.suptitle(title, y=0.995, va='top')
    renderer = fig.canvas.get_renderer()
    fig_w = fig.bbox.width * (1 - 2 * margin)
    for _ in range(6):
        w = t.get_window_extent(renderer).width
        if w <= fig_w:
            break
        longest = max(len(line) for line in t.get_text().split('\n'))
        t.set_text(textwrap.fill(title, max(10, int(longest * fig_w / w) - 1)))
    h = t.get_window_extent(renderer).height / fig.bbox.height
    fig.tight_layout(rect=(0, 0, 1, 1 - h - 0.012))


def summarize_and_plot(result, p_idx_pairs, labels, spec, png_path, title):
    """
    p_idx_pairs : list [(i,j), ...] chỉ số S để lấy (ví dụ (0,0)=S11, (1,0)=S21)
    labels      : tên tương ứng để in/vẽ, ví dụ ['S11','S21']
    spec        : dict {label: giới hạn}, None = bỏ qua. Giới hạn là một trong hai dạng:
                    (lo_db, hi_db)    - tuyệt đối, dùng chung cho mọi tần số;
                    ('nom', tol_db)   - TƯƠNG ĐỐI: danh định (mô phỏng không mismatch) ± tol_db,
                                        tính riêng cho từng tần số.
                  Trục x luôn bám theo dữ liệu (histogram + danh định); dải spec nằm ngoài vùng
                  này chỉ bị cắt, không còn kéo giãn trục làm histogram thu thành một vạch.
    """
    freqs_eval = result['freqs_eval']
    S_trial = result['S_trial']   # (n_trials, nf, N, N)
    S_nom = result['S_nom']       # (nf, N, N)
    n_trials = result['n_trials']

    n_freq = len(freqs_eval)
    n_metric = len(p_idx_pairs)
    # chiều rộng tối thiểu 6 inch để tiêu đề không bị cắt khi chỉ có 1 tần số khảo sát
    fig, axes = plt.subplots(n_metric, n_freq, figsize=(max(4.2 * n_freq, 6.0), 3.4 * n_metric),
                              squeeze=False)

    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(f"Số trial = {n_trials} | phân bố = GAUSS | dung sai = ±{result['tol_frac'] * 100:.1f}% "
          f"| thời gian = {result['elapsed']:.3f}s ({result['elapsed'] / n_trials * 1e3:.3f} ms/trial)")
    z0p = result.get('z0_ports')
    if z0p is not None:
        print("Z0 cổng ngoài = [" + ", ".join(_fmt_z(z) for z in z0p) + "] Ω"
              + ("" if any(z != 50 for z in z0p) else "  (mặc định)"))

    for mrow, ((pi, pj), lab) in enumerate(zip(p_idx_pairs, labels)):
        for kcol, fv in enumerate(freqs_eval):
            vals_db = _db(S_trial[:, kcol, pi, pj])
            nom_db = _db(S_nom[kcol, pi, pj])
            mean, std = vals_db.mean(), vals_db.std()
            p5, p50, p95 = np.percentile(vals_db, [5, 50, 95])

            msg = (f"  f={fv / 1e9:7.3f} GHz  {lab}: danh định={nom_db:7.3f} dB  "
                   f"mean={mean:7.3f}  std={std:6.3f}  "
                   f"P5/50/95={p5:7.3f}/{p50:7.3f}/{p95:7.3f} dB")
            if spec and lab in spec:
                lo, hi = _spec_band(spec[lab], nom_db)
                ok = (vals_db >= lo) & (vals_db <= hi)
                yield_pct = 100.0 * ok.mean()
                msg += f"  | spec[{lo:.3f},{hi:.3f}]dB -> yield={yield_pct:5.1f}%"
            print(msg)

            ax = axes[mrow][kcol]
            ax.hist(vals_db, bins=40, color='#3a7ca5', edgecolor='white', alpha=0.85)
            ax.axvline(nom_db, color='k', ls='--', lw=1.5, label=f'danh định={nom_db:.2f}dB')
            lo_x, hi_x = min(vals_db.min(), nom_db), max(vals_db.max(), nom_db)
            pad = 0.06 * (hi_x - lo_x) if hi_x > lo_x else 0.5
            if spec and lab in spec:
                lo, hi = _spec_band(spec[lab], nom_db)
                ax.axvspan(lo, hi, color='green', alpha=0.08)
            ax.set_xlim(lo_x - pad, hi_x + pad)       # bám theo dữ liệu, không để dải spec kéo giãn trục
            ax.set_title(f"{lab} @ {fv / 1e9:.3f} GHz", fontsize=10)
            ax.set_xlabel(f"{lab} (dB)")
            ax.set_ylabel("số trial")
            ax.legend(fontsize=8)

    _fit_suptitle(fig, title)
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"  -> đã lưu histogram: {png_path}")


# =====================================================================
# 4. TỰ KIỂM THỬ:  python3 monte_carlo.py --test
# =====================================================================
def self_test():
    import os
    import re
    import tempfile
    import warnings
    from components import renormalize_general

    here = os.path.dirname(os.path.abspath(__file__))
    old_cwd = os.getcwd()
    os.chdir(here)
    results = []
    ZC = {1: 75, 2: "50+10j"}
    Z_EXT = [75 + 0j, 50 + 10j]

    def check(name, ok, extra=""):
        results.append(bool(ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")

    def run(net, port_z0=None, n=40, seed=5, f=(0.5e9, 1.0e9), **kw):
        kw.setdefault("keys", ("R", "L", "C"))
        return run_monte_carlo(net, freqs_eval=list(f), tol_frac=0.05, n_trials=n, seed=seed,
                               port_z0=port_z0, **kw)

    try:
        # 1. mặc định
        a = run("filter_lc.net")
        b = run("filter_lc.net", port_z0={1: 50, 2: 50})
        check("port_z0=None == port_z0 {50,50} (bit-exact)",
              np.array_equal(a["S_trial"], b["S_trial"]) and np.array_equal(a["S_nom"], b["S_nom"]))
        check("z0_ports mặc định = [50, 50]", a["z0_ports"] == [50 + 0j, 50 + 0j])

        # 2. Z0 phức = chuẩn hóa lại kết quả 50 Ω của CHÍNH trial đó (cùng seed -> cùng bộ giá trị linh kiện)
        #    (kiểm tra nhất quán nội bộ; đối chiếu độc lập với scikit-rf ở mục 5)
        for nf, keys, f in [("filter_lc.net", ("R", "L", "C"), (0.5e9, 1.0e9)), ("amplifier.net", ("R",), (2.4e9,))]:
            a = run(nf, f=f, keys=keys)
            b = run(nf, ZC, f=f, keys=keys)
            worst = 0.0
            for arr50, arrz in [(a["S_trial"], b["S_trial"]), (a["S_nom"], b["S_nom"])]:
                for s50, sz in zip(arr50.reshape(-1, 2, 2), arrz.reshape(-1, 2, 2)):
                    worst = max(worst, np.abs(sz - renormalize_general(s50, 50, Z_EXT)).max())
            check(f"{nf}: MC với Z0 phức = chuẩn hóa lại MC 50 Ω", worst < 1e-11,
                  f"(sai khác {worst:.1e}, {a['n_trials']} trial)")
            check(f"{nf}: z0_ports ghi nhận đúng", b["z0_ports"] == Z_EXT)
            check(f"{nf}: Z0 phức làm S thay đổi thật", np.abs(a["S_trial"] - b["S_trial"]).max() > 1e-3)

        # 3. các cách khai báo Z0
        ref = run("filter_lc.net", ZC)
        lst = run("filter_lc.net", [75, 50 + 10j])
        check("port_z0 dạng list == dạng dict", np.array_equal(ref["S_trial"], lst["S_trial"]))
        txt = open("filter_lc.net", encoding="utf-8").read()
        txt = re.sub(r"(?m)^PORT 1 a[ \t]*\r?$", "PORT 1 a Z0=75", txt)
        txt = re.sub(r"(?m)^PORT 2 c[ \t]*\r?$", "PORT 2 c Z0=50+10j", txt)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lc_z.net")
            open(path, "w", encoding="utf-8").write(txt)
            inl = run(path)
        check("Z0 khai trong .net == port_z0 truyền qua tham số (bit-exact)",
              np.array_equal(ref["S_trial"], inl["S_trial"]) and inl["z0_ports"] == Z_EXT)
        part = run("filter_lc.net", {2: "50+10j"})
        check("chỉ khai 1 cổng: cổng còn lại giữ 50", part["z0_ports"] == [50 + 0j, 50 + 10j])

        # 4. giá trị sai
        ok = True
        for bad in [{3: 50}, {1: -5}, {1: 0}, {1: "abc"}, {1: "-5+3j"}]:
            try:
                run("filter_lc.net", bad, n=1)
                ok = False
            except NetlistError:
                pass
        check("port_z0 sai (cổng không tồn tại / Re<=0 / chuỗi rác) -> NetlistError", ok)

        # 5. đối chiếu scikit-rf (bỏ qua nếu chưa cài)
        try:
            import skrf as rf
            import compare_algorithms as ca
        except Exception as exc:
            print(f"[SKIP] đối chiếu scikit-rf: {type(exc).__name__}: {exc}")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for nf, f in [("filter_lc.net", [0.5e9, 1.0e9]), ("amplifier.net", [2.4e9])]:
                    keys = ("R", "L", "C") if nf == "filter_lc.net" else ("R",)
                    r = run(nf, ZC, n=8, f=f, keys=keys)
                    net = read_netlist(nf).set_port_z0(ZC)
                    circuit, perm = ca.build_skrf_circuit(net, np.array(f))
                    d = np.abs(r["S_nom"] - circuit.network.s[:, perm, :][:, :, perm]).max()
                    check(f"{nf}: S danh định (MC, Z0 phức) khớp scikit-rf {rf.__version__}", d < 1e-9,
                          f"(sai khác {d:.1e})")
                    rng = np.random.default_rng(5)       # dùng lại đúng dãy ngẫu nhiên của run_monte_carlo
                    worst = 0.0
                    for i in range(8):
                        nt = make_trial_netlist(read_netlist(nf).set_port_z0(ZC), 0.05, rng, keys=keys)
                        c_i, p_i = ca.build_skrf_circuit(nt, np.array(f))
                        worst = max(worst, np.abs(r["S_trial"][i] - c_i.network.s[:, p_i, :][:, :, p_i]).max())
                    check(f"{nf}: 8 trial ngẫu nhiên (Z0 phức) khớp scikit-rf (độc lập với solver)", worst < 1e-9,
                          f"(sai khác {worst:.1e})")
        # 6. spec tương đối / tuyệt đối và vẽ histogram
        check("_spec_band: ('nom', tol) = danh định ± tol; (lo, hi) giữ nguyên",
              _spec_band(('nom', 0.3), -0.5) == (-0.8, -0.2) and _spec_band((-60, -10), -9.6) == (-60.0, -10.0))
        try:
            _spec_band(('abc', 1), 0.0)
            ok = False
        except ValueError:
            ok = True
        check("_spec_band: kiểu lạ -> ValueError", ok)
        res = run("filter_lc.net", n=40)
        nom1 = _db(res['S_nom'][1, 1, 0])
        with tempfile.TemporaryDirectory() as tmp:
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                summarize_and_plot(res, [(0, 0), (1, 0)], ['S11', 'S21'],
                                   {'S11': ('nom', 3.0), 'S21': ('nom', 0.3)},
                                   os.path.join(tmp, "t.png"), "test")
            check("summarize_and_plot chạy với spec tương đối, in yield, ghi PNG",
                  os.path.exists(os.path.join(tmp, "t.png")) and "yield=" in buf.getvalue()
                  and f"{nom1 - 0.3:.3f}" in buf.getvalue())
    finally:
        os.chdir(old_cwd)

    print(f"\n=> {sum(results)}/{len(results)} PASS")
    return all(results)


# =====================================================================
# 5. CHẠY DEMO TRÊN 2 NETLIST CÓ SẴN (+ 2 trường hợp Z0 phức)
# =====================================================================
def demo():
    # --- 4.1 Lọc LC Chebyshev (filter_lc.net): nhiễu C/L ±5% (Gauss),
    #     khảo sát trong băng thông (500 MHz) và cạnh băng (1.0 GHz = fc) ---
    res_f = run_monte_carlo("filter_lc.net", freqs_eval=[0.5e9, 1.0e9],
                             tol_frac=0.05, n_trials=2000, seed=1)
    summarize_and_plot(
        res_f,
        p_idx_pairs=[(0, 0), (1, 0)], labels=['S11', 'S21'],
        spec={'S11': ('nom', 3.0), 'S21': ('nom', 0.3)},     # danh định ± 3 dB / ± 0.3 dB
        png_path="monte_carlo_filter_lc.png",
        title="Monte-Carlo filter_lc.net: dung sai C/L ±5% (Gauss, 2000 trial)")

    # --- 4.2 Tầng khuếch đại transistor (amplifier.net): nhiễu ĐIỆN TRỞ
    #     HỒI TIẾP RFB ±10% (Gauss) (linh kiện dễ dao động nhất vì tạo vòng
    #     kín quanh transistor), khảo sát tại tần số thiết kế 2.4 GHz ---
    res_a = run_monte_carlo("amplifier.net", freqs_eval=[2.4e9],
                             tol_frac=0.10, n_trials=2000, seed=2, keys=('R',))
    summarize_and_plot(
        res_a,
        p_idx_pairs=[(0, 0), (1, 0)], labels=['S11', 'S21'],
        spec={'S21': ('nom', 0.15)},     # danh định (5.72 dB) ± 0.15 dB
        png_path="monte_carlo_amplifier.png",
        title="Monte-Carlo amplifier.net: dung sai RFB ±10% (Gauss, 2000 trial)")

    # --- 4.3 Giống 4.1/4.2 nhưng cổng ngoài có Z0 PHỨC và KHÁC NHAU (không sửa file .net):
    #     P1 = 75 Ω, P2 = 50+10j Ω. Linh kiện vẫn nhiễu như cũ; chỉ tham chiếu cổng đổi. ---
    ZC = {1: 75, 2: "50+10j"}
    res_fz = run_monte_carlo("filter_lc.net", freqs_eval=[0.5e9, 1.0e9],
                              tol_frac=0.05, n_trials=2000, seed=1, port_z0=ZC)
    summarize_and_plot(
        res_fz,
        p_idx_pairs=[(0, 0), (1, 0)], labels=['S11', 'S21'],
        spec=None,
        png_path="monte_carlo_filter_lc_z0c.png",
        title="Monte-Carlo filter_lc.net, Z0 cổng = 75 / 50+10j Ω (Gauss, 2000 trial)")

    res_az = run_monte_carlo("amplifier.net", freqs_eval=[2.4e9],
                              tol_frac=0.10, n_trials=2000, seed=2, keys=('R',), port_z0=ZC)
    summarize_and_plot(
        res_az,
        p_idx_pairs=[(0, 0), (1, 0)], labels=['S11', 'S21'],
        spec=None,
        png_path="monte_carlo_amplifier_z0c.png",
        title="Monte-Carlo amplifier.net, Z0 cổng = 75 / 50+10j Ω (Gauss, 2000 trial)")

    print(f"\n{'=' * 78}\nHOÀN TẤT MONTE-CARLO.\n{'=' * 78}")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv[1:]:
        raise SystemExit(0 if self_test() else 1)
    demo()
