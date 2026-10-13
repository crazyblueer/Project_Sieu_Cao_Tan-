import numpy as np 

#PHASE 5: GHEP CONG HON HOP, ko con don gian chi cascade 2 cong lien tiep
def connect_ports(S, p, q):
    """
    Noi cong p voi cong q cua CUNG mot mang S (NxN).
    Tra ve ma tran S moi, nho hon 2 chieu (2 cong p,q bien mat).
    Cac cong con lai giu nguyen thu tu.
    """
    S = np.asarray(S, dtype=complex)
    N = S.shape[0]
    outside = [k for k in range(N) if k != p and k != q]   # cac cong ngoai con lai

    D = (1 - S[p,q])*(1 - S[q,p]) - S[p,p]*S[q,q]          # vong doi (mau so)

    M = len(outside)
    S_new = np.zeros((M, M), dtype=complex)
    for a, k in enumerate(outside):
        for b, l in enumerate(outside):
            S_new[a,b] = S[k,l] + (
                S[k,p]*S[q,l]*(1 - S[q,p])
              + S[k,q]*S[p,l]*(1 - S[p,q])
              + S[k,p]*S[p,l]*S[q,q]
              + S[k,q]*S[q,l]*S[p,p]
            ) / D
    return S_new

def block_diag(nets):
    """Gom cac khoi thanh 1 ma tran lon (cheo-khoi). nets: list of (S, [ten_cong])."""
    total = sum(S.shape[0] for S,_ in nets)
    big = np.zeros((total,total), dtype=complex); labels = []; i = 0
    for S, lab in nets:
        n = S.shape[0]; big[i:i+n, i:i+n] = S; labels += lab; i += n
    return big, labels


def connect_named(S, labels, name_p, name_q):
    """Noi 2 cong theo ten, dung connect_ports da co."""
    p = labels.index(name_p); q = labels.index(name_q)
    S_new = connect_ports(S, p, q)
    new_labels = [labels[k] for k in range(len(labels)) if k != p and k != q]
    return S_new, new_labels

