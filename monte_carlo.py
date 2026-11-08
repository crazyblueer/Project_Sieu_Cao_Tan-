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
    - Giải lại mạng bằng solve_network_global() (thuật toán connection-matrix,
      đã kiểm chứng khớp pairwise/scikit-rf tới 1e-6..1e-16, xem
      compare_algorithms.py) tại (các) tần số cần khảo sát.
    - Thu thập phân bố |S11|,|S21| (dB) trên toàn bộ trial -> thống kê
      mean/std/percentile + tỷ lệ ĐẠT spec (yield) -> dung sai linh kiện ảnh
      hưởng thế nào đến hiệu năng mạch thực tế (không chỉ giá trị lý tưởng).

Dùng lại TRỰC TIẾP các module đã có (không viết lại engine):
    netlist.py    -> đọc .net, build_network_at_freq()
    solver.py     -> solve_network_global() (connection-matrix, nhanh nhất)
    components.py -> to_si() để đổi giá trị '5.4303p' -> số thực trước khi nhiễu

Chạy trực tiếp: python3 monte_carlo.py
"""

import copy
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from netlist import read_netlist
from solver import solve_network_global
from components import to_si


# =====================================================================
# 1. NHIỄU NGẪU NHIÊN MỘT PHẦN TỬ NETLIST (PHÂN BỐ GAUSS)
# =====================================================================
def perturb_value(nominal_str, tol_frac, rng, sigma_rule=3.0, clip_sigma=None):
    """
    Nhiễu MỘT giá trị linh kiện (string SI, ví dụ '5.4303p') quanh giá trị
    danh định, theo PHÂN BỐ GAUSS (chuẩn):

        eps ~ N(0, sigma^2)   voi   sigma = tol_frac / sigma_rule

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


# =====================================================================
# 2. CHẠY N TRIAL TẠI (CÁC) TẦN SỐ KHẢO SÁT
# =====================================================================
def run_monte_carlo(netlist_path, freqs_eval, tol_frac=0.05, n_trials=2000,
                     z0=50.0, seed=123, keys=('R', 'L', 'C'), sigma_rule=3.0,
                     clip_sigma=None):
    """
    Dung sai được mô hình theo PHÂN BỐ GAUSS, sigma = tol_frac / sigma_rule.

    Trả về dict:
        'S_nom'   : S danh định tại mỗi tần số khảo sát, shape (nf, N, N)
        'S_trial' : S của tất cả trial, shape (n_trials, nf, N, N)
        'elapsed' : thời gian chạy (s)
        'ext'     : nhãn cổng ngoài (thứ tự P1,P2,...)
    """
    net = read_netlist(netlist_path)
    freqs_eval = np.atleast_1d(np.asarray(freqs_eval, dtype=float))
    rng = np.random.default_rng(seed)

    # --- danh định ---
    S_nom = []
    ext_ref = None
    for fv in freqs_eval:
        blocks, conns, ext = net.build_network_at_freq(fv, z0=z0)
        S_nom.append(solve_network_global(blocks, conns, ext))
        ext_ref = ext
    S_nom = np.array(S_nom)

    # --- các trial ---
    S_trial = np.empty((n_trials, len(freqs_eval)) + S_nom.shape[1:], dtype=complex)
    t0 = time.perf_counter()
    for i in range(n_trials):
        net_t = make_trial_netlist(net, tol_frac, rng, keys=keys,
                                    sigma_rule=sigma_rule, clip_sigma=clip_sigma)
        for k, fv in enumerate(freqs_eval):
            blocks, conns, ext = net_t.build_network_at_freq(fv, z0=z0)
            S_trial[i, k] = solve_network_global(blocks, conns, ext)
    elapsed = time.perf_counter() - t0

    return {'S_nom': S_nom, 'S_trial': S_trial, 'elapsed': elapsed,
            'ext': ext_ref, 'freqs_eval': freqs_eval, 'n_trials': n_trials,
            'tol_frac': tol_frac}


# =====================================================================
# 3. THỐNG KÊ + VẼ HISTOGRAM
# =====================================================================
def _db(x):
    return 20.0 * np.log10(np.abs(x) + 1e-300)


def summarize_and_plot(result, p_idx_pairs, labels, spec, png_path, title):
    """
    p_idx_pairs : list [(i,j), ...] chỉ số S để lấy (ví dụ (0,0)=S11, (1,0)=S21)
    labels      : tên tương ứng để in/vẽ, ví dụ ['S11','S21']
    spec        : dict {label: (lo_db, hi_db)} ngưỡng ĐẠT spec (dB), None = bỏ qua
    """
    freqs_eval = result['freqs_eval']
    S_trial = result['S_trial']   # (n_trials, nf, N, N)
    S_nom = result['S_nom']       # (nf, N, N)
    n_trials = result['n_trials']

    n_freq = len(freqs_eval)
    n_metric = len(p_idx_pairs)
    fig, axes = plt.subplots(n_metric, n_freq, figsize=(4.2 * n_freq, 3.4 * n_metric),
                              squeeze=False)

    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(f"Số trial = {n_trials} | phân bố = GAUSS | dung sai = ±{result['tol_frac'] * 100:.1f}% "
          f"| thời gian = {result['elapsed']:.3f}s ({result['elapsed'] / n_trials * 1e3:.3f} ms/trial)")

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
                lo, hi = spec[lab]
                ok = (vals_db >= lo) & (vals_db <= hi)
                yield_pct = 100.0 * ok.mean()
                msg += f"  | spec[{lo},{hi}]dB -> yield={yield_pct:5.1f}%"
            print(msg)

            ax = axes[mrow][kcol]
            ax.hist(vals_db, bins=40, color='#3a7ca5', edgecolor='white', alpha=0.85)
            ax.axvline(nom_db, color='k', ls='--', lw=1.5, label=f'danh định={nom_db:.2f}dB')
            if spec and lab in spec:
                lo, hi = spec[lab]
                ax.axvspan(lo, hi, color='green', alpha=0.08)
            ax.set_title(f"{lab} @ {fv / 1e9:.3f} GHz", fontsize=10)
            ax.set_xlabel(f"{lab} (dB)")
            ax.set_ylabel("số trial")
            ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"  -> đã lưu histogram: {png_path}")


# =====================================================================
# 4. CHẠY DEMO TRÊN 2 NETLIST CÓ SẴN
# =====================================================================
if __name__ == "__main__":
    # --- 4.1 Lọc LC Chebyshev (filter_lc.net): nhiễu C/L ±5% (Gauss),
    #     khảo sát trong băng thông (500 MHz) và cạnh băng (1.0 GHz = fc) ---
    res_f = run_monte_carlo("filter_lc.net", freqs_eval=[0.5e9, 1.0e9],
                             tol_frac=0.05, n_trials=2000, seed=1)
    summarize_and_plot(
        res_f,
        p_idx_pairs=[(0, 0), (1, 0)], labels=['S11', 'S21'],
        spec={'S11': (-60, -10), 'S21': (-3, 0.2)},
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
        spec={'S21': (10, 30)},
        png_path="monte_carlo_amplifier.png",
        title="Monte-Carlo amplifier.net: dung sai RFB ±10% (Gauss, 2000 trial)")

    print(f"\n{'=' * 78}\nHOÀN TẤT MONTE-CARLO.\n{'=' * 78}")
