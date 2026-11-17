import numpy as np

# ============================================================================
# ENGINE GHÉP MẠNG - HAI THUẬT TOÁN, MỘT GIAO DIỆN
#
#   connect(S, labels, name_pairs, z0_ports=None, method="matrix" | "pairwise")
#
#   method="matrix"   : ma trận kết nối (Monaco & Tiberio 1974) - nối TẤT CẢ các cặp cùng lúc
#                       bằng 1 phép nghịch đảo ma trận       -> connect_matrix()
#   method="pairwise" : nối dần từng cặp cổng (mỗi bước 1 phép nghịch đảo 2x2)
#                       -> connect_ports()
#
# Trở kháng tham chiếu Z0 của từng cổng LUÔN hỗ trợ PHỨC và KHÁC NHAU (sóng công suất,
# Kurokawa 1965). Không khai báo (z0_ports = None) => mọi cổng 50 ohm thực.
#
# Quy ước sóng CÔNG SUẤT (cùng định nghĩa mặc định s_def='power' của scikit-rf):
#       a = (V + Z I) / (2 sqrt(Re Z)),     b = (V - Z* I) / (2 sqrt(Re Z))
# Nối cổng p (tham chiếu Zp) với cổng q (tham chiếu Zq) = điều kiện vật lý
#       V_p = V_q,  I_p = -I_q
# tương đương a = Gc b (a, b là sóng vào / ra khỏi CÁC KHỐI, xem gamma_c).
# Khi Zp = Zq là số THỰC, Gc suy biến thành hoán vị [[0,1],[1,0]].
# ============================================================================

DEFAULT_Z0 = 50.0


def gamma_c(Zp, Zq):
    """
    Ma trận 2x2 của một cặp nối (p, q) khi trở kháng tham chiếu Zp, Zq có thể
    PHỨC và KHÁC NHAU (sóng công suất):

        [a_p]        1        [ Zq* - Zp       2 sqrt(Rp Rq) ] [b_p]
        [a_q] = ----------- * [ 2 sqrt(Rp Rq)  Zp* - Zq      ] [b_q]
                (Zp + Zq)*

    R = Re(Z), * = liên hợp. Suy ra từ V_p = V_q, I_p = -I_q.
    Tính chất: đối xứng, unitary; Zp = Zq thực => [[0,1],[1,0]].
    LƯU Ý: Gc KHÔNG phải phép đối hợp (Gc^-1 = Gc) trừ khi Zp = Zq thực.
    """
    Zp = complex(Zp)
    Zq = complex(Zq)
    r = 2.0 * np.sqrt(Zp.real * Zq.real)
    d = np.conj(Zp) + np.conj(Zq)
    return np.array([[np.conj(Zq) - Zp, r],
                     [r, np.conj(Zp) - Zq]], dtype=complex) / d


def _pair_gamma(Zp, Zq):
    """Gc của một cặp; nếu Zp = Zq là số thực thì trả về chính xác hoán vị [[0,1],[1,0]]."""
    Zp = complex(Zp)
    Zq = complex(Zq)
    if Zp == Zq and Zp.imag == 0.0:
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    return gamma_c(Zp, Zq)


# ============================================================================
# THUẬT TOÁN 1: NỐI DẦN TỪNG CẶP CỔNG (pairwise, Monaco & Tiberio 1974)
# ============================================================================

def connect_ports(S, p, q, Zp=DEFAULT_Z0, Zq=DEFAULT_Z0):
    """
    Nối cổng p (tham chiếu Zp) với cổng q (tham chiếu Zq) của CÙNG một mạng S (NxN).
    Zp, Zq có thể PHỨC và KHÁC NHAU (mặc định 50 ohm). Trả về S mới, nhỏ hơn 2 chiều
    (2 cổng p, q biến mất); các cổng còn lại giữ nguyên thứ tự và trở kháng tham chiếu.

        S' = S_EE + S_EP Gc (I - S_PP Gc)^-1 S_PE      (P = {p, q}, Gc = gamma_c(Zp, Zq))

    Mẫu số D = det(I - S_PP Gc). Khi Zp = Zq thực, D = (1-S_pq)(1-S_qp) - S_pp S_qq và
    cho kết quả trùng công thức từng phần tử kinh điển (k, l là 2 cổng còn lại bất kỳ):
        S'[k,l] = S[k,l] + ( S[k,p] S[q,l] (1 - S[p,q])
                           + S[k,q] S[p,l] (1 - S[q,p])
                           + S[k,p] S[p,l] S[q,q]
                           + S[k,q] S[q,l] S[p,p] ) / D
    """
    S = np.asarray(S, dtype=complex)
    N = S.shape[0]
    outside = [k for k in range(N) if k != p and k != q]
    P = [p, q]
    G = _pair_gamma(Zp, Zq)
    S_pp = S[np.ix_(P, P)]
    M = np.eye(2, dtype=complex) - S_pp @ G
    D = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]                       # nghịch đảo 2x2 bằng định thức
    Minv = np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]], dtype=complex) / D
    S_ee = S[np.ix_(outside, outside)]
    S_ep = S[np.ix_(outside, P)]
    S_pe = S[np.ix_(P, outside)]
    return S_ee + S_ep @ G @ Minv @ S_pe


def block_diag(nets):
    """Gộp các khối thành 1 ma trận lớn (chéo khối). nets: list of (S, [tên_cổng])."""
    total = sum(S.shape[0] for S, _ in nets)
    big = np.zeros((total, total), dtype=complex)
    labels = []
    i = 0
    for S, lab in nets:
        n = S.shape[0]
        big[i:i + n, i:i + n] = S
        labels += lab
        i += n
    return big, labels


# ============================================================================
# THUẬT TOÁN 2: MA TRẬN KẾT NỐI (global solve)
# ============================================================================

def connect_matrix(S, pairs, z0=None):
    """
    Thuật toán MA TRẬN KẾT NỐI (connection matrix / global solve).
    Nối TẤT CẢ các cặp cổng nội bộ trong 'pairs' CÙNG MỘT LÚC bằng 1 phép
    nghịch đảo ma trận duy nhất, thay vì lặp từng cặp như connect_ports().

    S     : ma trận S lớn (NxN), thường là kết quả block_diag() gộp nhiều khối.
    pairs : list[(p, q)] - danh sách các cặp chỉ số cổng cần nối trực tiếp.
            Mỗi cổng p, q là INDEX trong S (0-based), chỉ được xuất hiện
            trong ĐÚNG 1 cặp (không nối 1 cổng với 2 cổng khác).
    z0    : None (mọi cổng 50 ohm) hoặc list/array độ dài N - trở kháng tham chiếu
            (thực/phức, có thể khác nhau) của từng cổng theo INDEX trong S.

    Trả về: (S_total, outside) với
        S_total : ma trận S của mạng còn lại, đã LOẠI hết các cổng trong pairs
        outside : list index cổng GỐC (trong S đầu vào) theo thứ tự tương
                  ứng với hàng/cột của S_total

    Công thức:
        S_total = S_EE + S_EI @ Gamma @ inv(I - S_II @ Gamma) @ S_IE
    với E = các cổng ngoài (không trong pairs), I = các cổng nội bộ (trong pairs) và
    Gamma là ma trận khối-chéo các Gc = gamma_c(z0[p], z0[q]) của từng cặp (a_I = Gamma b_I).
    Các cổng ngoài giữ trở kháng tham chiếu của chúng.
    """
    S = np.asarray(S, dtype=complex)
    N = S.shape[0]
    if z0 is None:
        z0 = [DEFAULT_Z0] * N

    internal = sorted({p for pair in pairs for p in pair})
    seen = set()
    for p, q in pairs:
        if p == q:
            raise ValueError(f"Khong the tu noi cong {p} voi chinh no")
        if p in seen or q in seen:
            raise ValueError(f"Cong {p} hoac {q} xuat hien o nhieu hon 1 cap noi")
        seen.add(p)
        seen.add(q)

    outside = [k for k in range(N) if k not in internal]

    pos = {p: i for i, p in enumerate(internal)}
    K = len(internal)
    Gamma = np.zeros((K, K), dtype=complex)
    for p, q in pairs:
        G = _pair_gamma(z0[p], z0[q])     # a_p = G00 b_p + G01 b_q ; a_q = G10 b_p + G11 b_q
        Gamma[pos[p], pos[p]] = G[0, 0]
        Gamma[pos[p], pos[q]] = G[0, 1]
        Gamma[pos[q], pos[p]] = G[1, 0]
        Gamma[pos[q], pos[q]] = G[1, 1]

    S_ee = S[np.ix_(outside, outside)]
    S_ei = S[np.ix_(outside, internal)]
    S_ie = S[np.ix_(internal, outside)]
    S_ii = S[np.ix_(internal, internal)]

    Imat = np.eye(K, dtype=complex)
    S_total = S_ee + S_ei @ Gamma @ np.linalg.inv(Imat - S_ii @ Gamma) @ S_ie
    return S_total, outside


# ============================================================================
# GIAO DIỆN CHUNG THEO TÊN CỔNG: connect(..., method=)
# ============================================================================

METHODS = ("matrix", "pairwise")


def connect(S, labels, name_pairs, z0_ports=None, method="matrix"):
    """
    Nối các cặp cổng THEO TÊN, bằng thuật toán chọn qua 'method'.

    S          : ma trận S lớn (NxN), labels: list tên cổng theo thứ tự hàng/cột.
    name_pairs : list[(tên_p, tên_q)] - các cặp cổng cần nối.
    z0_ports   : None (mọi cổng 50 ohm) hoặc dict {tên cổng: trở kháng tham chiếu (thực/phức)};
                 cổng không có trong dict: 50 ohm. Cổng ngoài (không thuộc cặp nào) giữ nguyên.
    method     : "matrix"   -> connect_matrix() (nối hết cùng lúc, mặc định, nhanh)
                 "pairwise" -> connect_ports() lần lượt từng cặp

    Trả về (S_mới, labels_mới); labels_mới giữ thứ tự gốc của các cổng còn lại.
    """
    if method not in METHODS:
        raise ValueError(f"method phai la mot trong {METHODS}, nhan duoc {method!r}")
    S = np.asarray(S, dtype=complex)
    labels = list(labels)
    name_pairs = list(name_pairs)
    if not name_pairs:
        return S, labels

    def z(name):
        return DEFAULT_Z0 if z0_ports is None else z0_ports.get(name, DEFAULT_Z0)

    if method == "matrix":
        idx_pairs = [(labels.index(a), labels.index(b)) for a, b in name_pairs]
        S_total, outside = connect_matrix(S, idx_pairs, [z(l) for l in labels])
        return S_total, [labels[k] for k in outside]

    for a, b in name_pairs:
        p, q = labels.index(a), labels.index(b)
        S = connect_ports(S, p, q, z(a), z(b))
        labels = [labels[k] for k in range(len(labels)) if k != p and k != q]
    return S, labels


# ============================================================================
# TỰ KIỂM THỬ:  python engine.py
# ============================================================================

def _brute_force(S, pairs, z0):
    """
    Đáp án ĐỘC LẬP (không dùng công thức đóng): giải thẳng hệ tuyến tính của các sóng
        b = S a ;  a_p = Gc b (mỗi cặp nối) ;  a_ngoài = kích thích đơn vị
    rồi đọc S_mới = b_ngoài. Dùng để đối chiếu cả 2 thuật toán.
    """
    S = np.asarray(S, dtype=complex)
    N = S.shape[0]
    internal = sorted({p for pr in pairs for p in pr})
    outside = [k for k in range(N) if k not in internal]
    n_eq = N + len(internal) + len(outside)                 # ẩn số: [a(0..N-1), b(0..N-1)]
    A = np.zeros((n_eq, 2 * N), dtype=complex)
    row = 0
    for k in range(N):                                      # b - S a = 0
        A[row, N + k] = 1.0
        A[row, :N] -= S[k, :]
        row += 1
    for p, q in pairs:                                      # a_pq = Gc b_pq
        G = gamma_c(z0[p], z0[q])
        for i, u in enumerate((p, q)):
            A[row, u] = 1.0
            A[row, N + p] -= G[i, 0]
            A[row, N + q] -= G[i, 1]
            row += 1
    ext_rows = {}
    for j in outside:                                       # a_j = kích thích
        A[row, j] = 1.0
        ext_rows[j] = row
        row += 1
    out = np.zeros((len(outside), len(outside)), dtype=complex)
    for c, j in enumerate(outside):
        rhs = np.zeros(n_eq, dtype=complex)
        rhs[ext_rows[j]] = 1.0
        x = np.linalg.solve(A, rhs)
        out[:, c] = x[N + np.array(outside)]
    return out, outside


def self_test():
    from components import renormalize_general
    results = []

    def check(name, ok, extra=""):
        results.append(bool(ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")

    TOL = 1e-9

    # ---- 1. gamma_c
    swap = np.array([[0, 1], [1, 0]], dtype=complex)
    check("gamma_c(50,50) = hoan vi [[0,1],[1,0]]", np.array_equal(gamma_c(50, 50), swap))
    check("gamma_c(75,75) = hoan vi", np.allclose(gamma_c(75, 75), swap, atol=1e-15))
    check("gamma_c(50,75) = [[0.2,0.9798],[0.9798,-0.2]]",
          np.allclose(gamma_c(50, 75), [[0.2, np.sqrt(0.96)], [np.sqrt(0.96), -0.2]], atol=1e-12))
    rng = np.random.default_rng(0)
    ok_sym = ok_uni = ok_phys = True
    for _ in range(200):
        zp = complex(rng.uniform(5, 200), rng.uniform(-100, 100))
        zq = complex(rng.uniform(5, 200), rng.uniform(-100, 100))
        G = gamma_c(zp, zq)
        ok_sym &= np.allclose(G, G.T, atol=1e-13)
        ok_uni &= np.allclose(G.conj().T @ G, np.eye(2), atol=1e-12)
        V, I = complex(rng.normal(), rng.normal()), complex(rng.normal(), rng.normal())
        a = np.array([(V + zp * I) / (2 * np.sqrt(zp.real)), (V + zq * (-I)) / (2 * np.sqrt(zq.real))])
        b = np.array([(V - np.conj(zp) * I) / (2 * np.sqrt(zp.real)), (V - np.conj(zq) * (-I)) / (2 * np.sqrt(zq.real))])
        ok_phys &= np.allclose(G @ b, a, atol=1e-12)
    check("gamma_c doi xung (200 cap Z0 phuc ngau nhien)", ok_sym)
    check("gamma_c unitary (200 cap)", ok_uni)
    check("gamma_c thoa V_p=V_q, I_p=-I_q (a = Gc b, 200 cap)", ok_phys)
    zp, zq = 50 + 20j, 75 - 10j
    check("gamma_c KHONG la phep doi hop khi Z0 phuc/khac nhau (Gc@Gc != I)",
          not np.allclose(gamma_c(zp, zq) @ gamma_c(zp, zq), np.eye(2)))

    # ---- 2. 2 thuật toán trên ma trận S ngẫu nhiên, Z0 phức / khác nhau
    n = 8
    S = 0.35 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(n)
    z = [complex(rng.uniform(15, 150), rng.uniform(-60, 60)) for _ in range(n)]
    pairs = [(0, 3), (1, 5), (2, 6)]                       # cổng ngoài: 4, 7
    labels = [f"p{k}" for k in range(n)]
    zd = dict(zip(labels, z))
    nm = [(labels[a], labels[b]) for a, b in pairs]

    brute, outs = _brute_force(S, pairs, z)
    got, outs2 = connect_matrix(S, pairs, z)
    check("connect_matrix(Z0 phuc/khac nhau) = giai truc tiep he song (doc lap)",
          outs == outs2 and np.abs(got - brute).max() < TOL, f"(sai khac {np.abs(got - brute).max():.1e})")
    S50 = renormalize_general(S, z, 50.0)
    S_c, _ = connect_matrix(S50, pairs)
    truth = renormalize_general(S_c, 50.0, [z[k] for k in outs])
    check("connect_matrix = chuan hoa ve 50 ohm -> noi -> chuan hoa nguoc",
          np.abs(got - truth).max() < TOL, f"(sai khac {np.abs(got - truth).max():.1e})")
    Sp, lab_p = connect(S, labels, nm, zd, method="pairwise")
    Sm, lab_m = connect(S, labels, nm, zd, method="matrix")
    check("connect(method=pairwise) = connect(method=matrix), Z0 phuc",
          lab_p == lab_m and np.abs(Sp - Sm).max() < TOL, f"(sai khac {np.abs(Sp - Sm).max():.1e})")
    check("connect(method=pairwise) = giai truc tiep he song", np.abs(Sp - brute).max() < TOL)

    # định thức: det(I - S_II Gamma) = tích các D của từng bước nối cặp
    internal = sorted({p for pr in pairs for p in pr})
    pos = {p: i for i, p in enumerate(internal)}
    Gam = np.zeros((len(internal),) * 2, dtype=complex)
    for p, q in pairs:
        Gc = gamma_c(z[p], z[q])
        for (u, i1) in ((p, 0), (q, 1)):
            for (v, i2) in ((p, 0), (q, 1)):
                Gam[pos[u], pos[v]] = Gc[i1, i2]
    detG = np.linalg.det(np.eye(len(internal)) - S[np.ix_(internal, internal)] @ Gam)
    prodD, Sx, labx = 1.0, S.copy(), labels[:]
    for a, b in pairs:
        p, q = labx.index(labels[a]), labx.index(labels[b])
        Mx = np.eye(2) - Sx[np.ix_([p, q], [p, q])] @ gamma_c(zd[labels[a]], zd[labels[b]])
        prodD *= np.linalg.det(Mx)
        Sx, labx = connect(Sx, labx, [(labels[a], labels[b])], zd, method="pairwise")
    check("det(I - S_II Gamma_c) = tich cac D cua cac buoc noi (2 thuat toan nhat quan)",
          abs(detG - prodD) < 1e-10 * max(1, abs(detG)))

    # ---- 3. Z0 mặc định và Z0 thực đều
    Sd1, _ = connect(S, labels, nm, None, method="matrix")
    Sd2, _ = connect(S, labels, nm, {l: 50.0 for l in labels}, method="matrix")
    check("z0_ports = None  ==  moi cong 50 ohm (bit-exact)", np.array_equal(Sd1, Sd2))
    Sd3, _ = connect(S, labels, nm, None, method="pairwise")
    check("pairwise mac dinh = matrix mac dinh", np.abs(Sd1 - Sd3).max() < 1e-13)
    z75 = {l: 75.0 for l in labels}
    A75, _ = connect(S, labels, nm, z75, method="matrix")
    check("Z0 thuc deu 75 cho ket qua nhu 50 deu (cong thuc noi khong phu thuoc Z0 chung)",
          np.abs(A75 - Sd1).max() < 1e-13)

    # ---- 4. công thức từng phần tử kinh điển (Z0 thực bằng nhau) và mạng KHÔNG thuận nghịch
    def classic(S, p, q):
        N = S.shape[0]
        out = [k for k in range(N) if k not in (p, q)]
        D = (1 - S[p, q]) * (1 - S[q, p]) - S[p, p] * S[q, q]
        R = np.zeros((len(out), len(out)), dtype=complex)
        for a, k in enumerate(out):
            for b, m in enumerate(out):
                R[a, b] = S[k, m] + (S[k, p] * S[q, m] * (1 - S[p, q]) + S[k, q] * S[p, m] * (1 - S[q, p])
                                     + S[k, p] * S[p, m] * S[q, q] + S[k, q] * S[q, m] * S[p, p]) / D
        return R

    Sr = 0.4 * (rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))
    check("connect_ports(Zp=Zq=50) = cong thuc kinh dien",
          np.abs(connect_ports(Sr, 1, 4) - classic(Sr, 1, 4)).max() < 1e-13)
    # transistor: S12 = 0.02 << S21 = 3.0 (không thuận nghịch), + cổng ngoài ghép bất đối xứng
    T = np.zeros((3, 3), dtype=complex)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = 0.3 - 0.2j, 0.02j, 3.0 - 1.0j, 0.4 + 0.1j
    T[0, 2], T[2, 0], T[1, 2], T[2, 1], T[2, 2] = 0.1, 0.5j, 0.6, -0.2, 0.1j
    br, _ = _brute_force(T, [(0, 1)], [50, 50, 50])
    check("mang KHONG thuan nghich (S12 << S21): pairwise = giai truc tiep he song",
          np.abs(connect_ports(T, 0, 1) - br).max() < 1e-13, f"(|S21/S12| = {abs(T[1, 0] / T[0, 1]):.0f})")
    check("mang KHONG thuan nghich: connection-matrix = giai truc tiep he song",
          np.abs(connect_matrix(T, [(0, 1)])[0] - br).max() < 1e-13)

    # ---- 5. lỗi đầu vào
    for what, fn in [("method sai", lambda: connect(S, labels, nm, None, method="abc")),
                     ("tu noi cong", lambda: connect_matrix(S, [(1, 1)])),
                     ("cong thuoc 2 cap", lambda: connect_matrix(S, [(0, 1), (1, 2)]))]:
        try:
            fn()
            ok = False
        except ValueError:
            ok = True
        check(f"{what} -> ValueError", ok)

    print(f"\n=> {sum(results)}/{len(results)} PASS")
    return all(results)


if __name__ == "__main__":
    raise SystemExit(0 if self_test() else 1)
