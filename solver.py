import numpy as np
from components import is_lossless, is_reciprocal, is_passive
from engine import connect_named, block_diag
from netlist import read_netlist 


def solve_network(blocks, connections, ext_port_labels):
    """Noi output cua netlist (B) bang engine ghep (connect_ports)."""
    big, lab = block_diag(blocks)          # EXT labels KHONG nam trong day

    ext_set = set(ext_port_labels)
    internal = []          # cap noi that (ca hai pin that)
    ext_map = {}           # EXT_Pk -> ten pin that duoc phoi ra ngoai
    for a, b in connections:
        a_ext, b_ext = a in ext_set, b in ext_set
        if a_ext and not b_ext:   ext_map[a] = b
        elif b_ext and not a_ext: ext_map[b] = a
        elif not a_ext and not b_ext: internal.append((a, b))
        else: raise ValueError(f"Hai cong ngoai noi truc tiep: {a},{b}")

    # thuc hien cac noi that, tieu thu tung cap
    S = big
    for a, b in internal:
        S, lab = connect_named(S, lab, a, b)

    # con lai = cac pin duoc chi dinh ngoai; sap theo thu tu port P1,P2,P3...
    ordered = [ext_map[e] for e in ext_port_labels]
    assert sorted(lab) == sorted(ordered), f"Con thua/thieu cong: {lab} vs {ordered}"
    idx = [lab.index(p) for p in ordered]
    return S[np.ix_(idx, idx)]

if __name__ == "__main__":
    net = read_netlist("wilkinson.net")
    blocks, connections, ext = net.build_network_at_freq(2.4e9, z0=50.0)
    S = solve_network(blocks, connections, ext)
    print("Cong con lai:", ext)
    print(np.round(S, 4))
    
    print(is_reciprocal(S), is_passive(S), is_lossless(S))
