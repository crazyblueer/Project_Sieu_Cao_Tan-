import numpy as np

# ============================================================================
# PHASE 5: GHEP CONG HON HOP (thuat toan "noi dan tung cap cong",
#          Monaco & Tiberio 1974) - ko con don gian chi cascade 2 cong lien tiep
# ============================================================================

def connect_ports(S, p, q):
    """
    BAN GOC (CO BUG) cua thuat toan "noi dan tung cap cong".

    Noi cong p voi cong q cua CUNG mot mang S (NxN).
    Tra ve ma tran S moi, nho hon 2 chieu (2 cong p,q bien mat).
    Cac cong con lai giu nguyen thu tu.

    Cong thuc DUNG (k, l la 2 cong con lai bat ky):
        D       = (1 - S[p,q])(1 - S[q,p]) - S[p,p] S[q,q]
        S'[k,l] = S[k,l] + ( S[k,p] S[q,l] (1 - S[p,q])
                           + S[k,q] S[p,l] (1 - S[q,p])
                           + S[k,p] S[p,l] S[q,q]
                           + S[k,q] S[q,l] S[p,p] ) / D

    BUG: 2 he so (1 - S[p,q]) va (1 - S[q,p])
    bi DAO cho nhau giua 2 so hang tu cua tu so (xem 2 dong danh dau "DAO (bug)"
    duoi day). Voi mang THUAN NGHICH (S[p,q] = S[q,p], vi du TLINE/RES/LC) hai
    so bang nhau nen loi bi che hoan toan; chi khi noi hai cong cua mot khoi
    KHONG thuan nghich (transistor: S12 != S21) ket qua moi lech so voi ban
    da sua (xem connect_ports_fixed) - do la cach loi duoc phat hien
    (xem transistor_bug_demo.py).

    GIU LAI HAM NAY de doi chieu/minh hoa loi - KHONG dung cho san xuat.
    San xuat (solve_network qua connect_named) dung connect_ports_fixed().
    """
    S = np.asarray(S, dtype=complex)
    N = S.shape[0]
    outside = [k for k in range(N) if k != p and k != q]   # cac cong ngoai con lai

    D = (1 - S[p,q])*(1 - S[q,p]) - S[p,p]*S[q,q]          # vong doi (mau so)

    M = len(outside)
    S_new = np.zeros((M, M), dtype=complex)
    for a, k in enumerate(outside):
        for b, m in enumerate(outside):
            S_new[a,b] = S[k,m] + (
                S[k,p]*S[q,m]*(1 - S[q,p])     # DAO (bug): phai la (1 - S[p,q])
              + S[k,q]*S[p,m]*(1 - S[p,q])     # DAO (bug): phai la (1 - S[q,p])
              + S[k,p]*S[p,m]*S[q,q]
              + S[k,q]*S[q,m]*S[p,p]
            ) / D
    return S_new


def connect_ports_fixed(S, p, q):
    """
    BAN DA SUA cua thuat toan "noi dan tung cap cong"
    (Monaco & Tiberio 1974).

    Noi cong p voi cong q cua CUNG mot mang S (NxN). Tra ve ma tran S moi,
    nho hon 2 chieu (2 cong p,q bien mat). Cac cong con lai giu nguyen thu tu.

    Cong thuc (k, l la 2 cong con lai bat ky):
        D       = (1 - S[p,q])(1 - S[q,p]) - S[p,p] S[q,q]
        S'[k,l] = S[k,l] + ( S[k,p] S[q,l] (1 - S[p,q])
                           + S[k,q] S[p,l] (1 - S[q,p])
                           + S[k,p] S[p,l] S[q,q]
                           + S[k,q] S[q,l] S[p,p] ) / D

    He so (1 - S[p,q]) di CUNG so hang S[k,p]*S[q,l] (duong di k -> p, roi
    q -> l), va (1 - S[q,p]) di cung S[k,q]*S[p,l].
    """
    S = np.asarray(S, dtype=complex)
    N = S.shape[0]
    outside = [k for k in range(N) if k != p and k != q]   # cac cong ngoai con lai

    D = (1 - S[p,q])*(1 - S[q,p]) - S[p,p]*S[q,q]          # vong doi (mau so)

    M = len(outside)
    S_new = np.zeros((M, M), dtype=complex)
    for a, k in enumerate(outside):
        for b, m in enumerate(outside):
            S_new[a,b] = S[k,m] + (
                S[k,p]*S[q,m]*(1 - S[p,q])
              + S[k,q]*S[p,m]*(1 - S[q,p])
              + S[k,p]*S[p,m]*S[q,q]
              + S[k,q]*S[q,m]*S[p,p]
            ) / D
    return S_new


def block_diag(nets):
    """Gom cac khoi thanh 1 ma tran lon (cheo-khoi). nets: list of (S, [ten_cong])."""
    total = sum(S.shape[0] for S,_ in nets)
    big = np.zeros((total,total), dtype=complex)
    labels = []
    i = 0
    for S, lab in nets:
        n = S.shape[0]
        big[i:i+n, i:i+n] = S
        labels += lab
        i += n
    return big, labels


def connect_named(S, labels, name_p, name_q):
    """Noi 2 cong theo ten, dung connect_ports_fixed() (ban DA SUA)."""
    p = labels.index(name_p)
    q = labels.index(name_q)
    S_new = connect_ports_fixed(S, p, q)
    new_labels = [labels[k] for k in range(len(labels)) if k != p and k != q]
    return S_new, new_labels


def connect_named_buggy(S, labels, name_p, name_q):
    """Nhu connect_named() nhung dung connect_ports() ban GOC co bug (de doi chieu/demo)."""
    p = labels.index(name_p)
    q = labels.index(name_q)
    S_new = connect_ports(S, p, q)
    new_labels = [labels[k] for k in range(len(labels)) if k != p and k != q]
    return S_new, new_labels


# ============================================================================
# THUAT TOAN MA TRAN KET NOI (global solve)
# Cac ham connect_matrix/connect_named_matrix (nguyen ly ghep cong) nam o day;
# solve_network_global (tich hop netlist + engine, cung vai tro voi
# solve_network) nam trong solver.py - xem solver.py.
# ============================================================================

def connect_matrix(S, pairs):
    """
    Thuat toan MA TRAN KET NOI (connection matrix / global solve).
    Noi TAT CA cac cap cong noi bo trong 'pairs' CUNG MOT LUC bang 1 phep
    nghich dao ma tran duy nhat, thay vi lap tung cap nhu connect_ports_fixed().

    S     : ma tran S lon (NxN), thuong la ket qua block_diag() gop nhieu khoi.
    pairs : list[(p, q)] - danh sach cac cap chi so cong can noi truc tiep.
            Moi cong p, q la INDEX trong S (0-based), chi duoc xuat hien
            trong DUNG 1 cap (khong noi 1 cong voi 2 cong khac).

    Tra ve: (S_total, outside) voi
        S_total : ma tran S cua mang con lai, da LOAI het cac cong trong pairs
        outside : list index cong GOC (trong S dau vao) theo thu tu tuong
                  ung voi hang/cot cua S_total (giong 'outside' cua
                  connect_ports_fixed).

    Cong thuc (xem ly_thuyet_ma_tran_ket_noi.md):
        S_total = S_EE + S_EI @ Gamma @ inv(I - S_II @ Gamma) @ S_IE
    voi Gamma la ma tran hoan vi doi xung ma hoa cac cap noi,
    E = cac cong ngoai (khong trong pairs), I = cac cong noi bo (trong pairs).
    """
    S = np.asarray(S, dtype=complex)
    N = S.shape[0]

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
        Gamma[pos[p], pos[q]] = 1.0
        Gamma[pos[q], pos[p]] = 1.0

    S_ee = S[np.ix_(outside, outside)]
    S_ei = S[np.ix_(outside, internal)]
    S_ie = S[np.ix_(internal, outside)]
    S_ii = S[np.ix_(internal, internal)]

    Imat = np.eye(K, dtype=complex)
    S_total = S_ee + S_ei @ Gamma @ np.linalg.inv(Imat - S_ii @ Gamma) @ S_ie
    return S_total, outside


def connect_named_matrix(S, labels, name_pairs):
    """
    Ban theo TEN cong (tuong tu connect_named() o tren) nhung dung
    thuat toan ma tran ket noi: noi NHIEU cap cung luc.

    name_pairs : list[(ten_cong_p, ten_cong_q)]
    """
    idx_pairs = [(labels.index(a), labels.index(b)) for a, b in name_pairs]
    S_total, outside = connect_matrix(S, idx_pairs)
    new_labels = [labels[k] for k in outside]
    return S_total, new_labels


if __name__ == "__main__":
    from netlist import read_netlist
    from solver import solve_network, solve_network_global
    from components import is_reciprocal, is_passive, is_lossless

    net = read_netlist("wilkinson.net")
    blocks, connections, ext = net.build_network_at_freq(2.4e9, z0=50.0)

    S_pairwise = solve_network(blocks, connections, ext)
    S_global = solve_network_global(blocks, connections, ext)

    print("Cong con lai:", ext)
    print("\n--- Pairwise (noi dan tung cap, solver.py -> connect_ports_fixed) ---")
    print(np.round(S_pairwise, 6))
    print("\n--- Connection-matrix (giai 1 lan, cung module nay) ---")
    print(np.round(S_global, 6))

    max_diff = np.max(np.abs(S_pairwise - S_global))
    print(f"\nMax abs diff giua 2 thuat toan: {max_diff:.3e}")
    print("KHOP" if max_diff < 1e-9 else "KHONG KHOP")

    rec, pas, los = is_reciprocal(S_global), is_passive(S_global), is_lossless(S_global)
    print(f"reciprocal/passive/lossless (connection-matrix): {rec}/{pas}/{los}  (mong doi True/True/False)")
