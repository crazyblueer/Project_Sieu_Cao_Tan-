"""
main.py — Kiem thu engine mo phong mach (topic 7).
Chay: python main.py
Yeu cau co san trong cung thu muc: wilkinson.net, test.s2p
"""

import numpy as np
import skrf as rf

from components import (
    tline_ideal, renormalize, renormalize_general,
    is_reciprocal, is_lossless, is_passive,
)
from netlist import read_netlist
from solver import solve_network
from touchstone import load_s2p


def test_renormalization():
    print("\n=== TEST 1: RENORMALIZATION (doi tro khang tham chieu) ===")
    S_old = tline_ideal(np.pi / 2)   # doan lambda/4, he tham chieu = Zc cua no

    # (a) Tuong thich nguoc: Z0 thuc -> ban tong quat phai trung ban cu
    a = np.allclose(renormalize_general(S_old, 50 * np.sqrt(2), 50),
                    renormalize(S_old, 50 * np.sqrt(2), 50))

    # (b) Sanity: doi ve chinh no khong doi gi; di roi ve -> nguyen ban
    z = [50 + 10j, 50 + 10j]
    b1 = np.allclose(renormalize_general(S_old, z, z), S_old)
    b2 = np.allclose(renormalize_general(renormalize_general(S_old, 50, z), z, 50), S_old)

    # (c) Cross-check scikit-rf (power wave) voi Z0 phuc
    ntwk = rf.Network(frequency=rf.Frequency(2.4, 2.4, 1, 'ghz'),
                      s=S_old.reshape(1, 2, 2), z0=50 * np.sqrt(2))
    ntwk.renormalize([50 + 10j, 50 + 10j], s_def='power')
    mine = renormalize_general(S_old, 50 * np.sqrt(2), [50 + 10j, 50 + 10j])
    c = np.allclose(ntwk.s[0], mine)

    print(f"  (a) tuong thich nguoc (Z0 thuc) : {_verdict(a)}")
    print(f"  (b) sanity (doi ve chinh no)    : {_verdict(b1 and b2)}")
    print(f"  (c) cross-check scikit-rf (phuc): {_verdict(c)}")
    return a and b1 and b2 and c


def test_netlist_wilkinson():
    print("\n=== TEST 2: NETLIST -> ENGINE (bo chia Wilkinson) ===")
    net = read_netlist("wilkinson.net")
    blocks, connections, ext = net.build_network_at_freq(2.4e9, z0=50.0)
    S = solve_network(blocks, connections, ext)

    print("  Cong ngoai:", ext)
    print("  Ma tran S:")
    print(_indent(np.round(S, 4)))

    # So voi ly thuyet Wilkinson: S21=S31=-0.707j, S11=S23=0
    expected = np.array([[0, -0.7071j, -0.7071j],
                         [-0.7071j, 0, 0],
                         [-0.7071j, 0, 0]])
    khop = np.allclose(S, expected, atol=1e-3)
    print(f"  Khop ly thuyet Wilkinson        : {_verdict(khop)}")

    rec, pas, los = is_reciprocal(S), is_passive(S), is_lossless(S)
    print(f"  reciprocal/passive/lossless     : {rec}/{pas}/{los}  (mong doi True/True/False)")
    return khop and rec and pas and (not los)


def test_touchstone():
    print("\n=== TEST 3: DOC FILE TOUCHSTONE (.s2p) ===")
    ts = load_s2p("test.s2p")
    S = ts.s_interp(2.4e9)
    print("  Ma tran S doc tu file:")
    print(_indent(np.round(S, 4)))

    expected = np.array([[0.3333, -0.9428j],
                         [-0.9428j, 0.3333]])
    khop = np.allclose(S, expected, atol=1e-3)
    print(f"  Khop gia tri mong doi            : {_verdict(khop)}")
    return khop


def _verdict(ok):
    return "PASS" if ok else "FAIL"


def _indent(arr):
    return "\n".join("    " + line for line in str(arr).splitlines())


if __name__ == "__main__":
    results = {
        "Renormalization": test_renormalization(),
        "Netlist -> Wilkinson": test_netlist_wilkinson(),
        "Touchstone .s2p": test_touchstone(),
    }
    print("\n=== TONG KET ===")
    for ten, ok in results.items():
        print(f"  {ten:24s}: {_verdict(ok)}")
