"""
Vi du minh hoa: pairwise connect_ports() dang bi loi dao he so (1-S[p,q]) <-> (1-S[q,p])
khi NOI 2 CHAN CUA 1 TRANSISTOR (mang KHONG thuan nghich, S12 != S21) voi mot mang
ben ngoai (vi du mot mach hoi tiep / tai).

Tai sao Wilkinson (TLINE, RES) khong phat hien duoc loi: moi khoi deu THUAN NGHICH
(S12 == S21 tai moi cap cong), nen dao (1-S[p,q]) <-> (1-S[q,p]) la dao 2 SO BANG NHAU
-> ket qua khong doi. Transistor pha vo dieu kien nay.
"""
import numpy as np
from engine import connect_matrix, connect_ports, connect_ports_fixed


def build_3port_with_transistor():
    """
    Mang 3 cong TRUC TIEP (khong qua block_diag, de p,q nam ngay trong cung 1 ma tran):
      cong 0 = IN  cua transistor
      cong 1 = OUT cua transistor
      cong 2 = cong ngoai con lai (vi du noi toi mot tai/hoi tiep ben ngoai)

    S cua transistor chiem khoi [0:2,0:2]: S01 = S12_transistor (~0.02, khuech dai
    NGUOC rat yeu), S10 = S21_transistor (~3.0, khuech dai THUAN manh) -> S01 != S10
    ro ret (|S10|/|S01| ~ 150x), dung dac trung vat ly cua BJT/FET khuech dai.

    Cong 2 duoc ghep hoi tiep nho vao ca hai chan transistor. QUAN TRONG: coupling
    nay phai BAT DOI XUNG (S02 != S20, S12c != S21c) - neu THUAN NGHICH (S02=S20,
    S12c=S21c) thi phan khac biet do bug gay ra se TRIET TIEU het cho mang 1 cong
    ngoai (da chung minh bang dai so: voi outside=[2] duy nhat, so hang them vao la
    (S[2,p]*S[q,2]-S[2,q]*S[p,2]) * (S[p,q]-S[q,p]) / D, thua so dau = 0 neu coupling
    thuan nghich doi xung -> dao he so khong gay sai lech gi, loi van an duoc!).
    """
    S11_tr = 0.3 * np.exp(1j * np.deg2rad(-40))
    S21_tr = 3.0 * np.exp(1j * np.deg2rad(60))     # IN->OUT: khuech dai THUAN, manh
    S12_tr = 0.02 * np.exp(1j * np.deg2rad(-100))  # OUT->IN: khuech dai NGUOC, rat yeu
    S22_tr = 0.4 * np.exp(1j * np.deg2rad(-20))

    S02 = 0.05 * np.exp(1j * np.deg2rad(10))    # IN(0) -> cong 2
    S20 = 0.05 * np.exp(1j * np.deg2rad(70))    # cong 2 -> IN(0): GOC KHAC S02 -> bat doi xung
    S12c = 0.04 * np.exp(1j * np.deg2rad(-30))  # OUT(1) -> cong 2
    S21c = 0.04 * np.exp(1j * np.deg2rad(50))   # cong 2 -> OUT(1): GOC KHAC S12c -> bat doi xung

    S = np.array([
        [S11_tr, S12_tr, S02],
        [S21_tr, S22_tr, S12c],
        [S20,    S21c,   0.1 + 0.05j],   # S22 cua cong 2 (phan xa nho)
    ], dtype=complex)
    return S


if __name__ == "__main__":
    S = build_3port_with_transistor()
    print("=== S goc 3 cong (cong 0=IN, cong 1=OUT cua transistor, cong 2=ngoai) ===")
    print(np.round(S, 6))

    p, q = 0, 1  # DUNG 2 chan IN/OUT cua transistor -> S[p,q] != S[q,p] that
    print(f"\nKiem tra S[{p},{q}] (=S12_transistor) va S[{q},{p}] (=S21_transistor):")
    print(f"  S[{p},{q}] = {S[p,q]:.4f}   S[{q},{p}] = {S[q,p]:.4f}   "
          f"(|S21|/|S12| = {abs(S[q,p])/abs(S[p,q]):.1f}x -> RO RET S[p,q] != S[q,p])")

    # --- (A) Pairwise hien tai trong engine.py (co bug dao he so) ---
    S_pairwise_buggy = connect_ports(S, p, q)

    # --- (B) Connection-matrix (ground truth, dan tu nguyen ly vat ly, da verify bang sympy) ---
    S_ground_truth, _ = connect_matrix(S, [(p, q)])

    print(f"\n=== Noi cong {p} (IN) voi cong {q} (OUT) -> chi con lai cong 2 ===")
    print("\n--- (A) pairwise connect_ports() HIEN TAI (engine.py, co bug) ---")
    print(np.round(S_pairwise_buggy, 6))

    print("\n--- (B) connection-matrix connect_matrix() (ground truth, da verify sympy) ---")
    print(np.round(S_ground_truth, 6))

    diff = np.abs(S_pairwise_buggy - S_ground_truth)
    print(f"\nMax abs diff (A vs B): {diff.max():.6f}")
    print("=> LECH LON, KHONG PHAI SAI SO LAM TRON (loi lo ro)" if diff.max() > 1e-6 else "=> KHOP (khong du lo loi)")

    # --- (C) Ban SUA LOI: goi THANG connect_ports_fixed() co san trong engine.py
    #     (dao dung vi tri 2 he so (1-S[p,q]) <-> (1-S[q,p]) so voi connect_ports() goc) ---
    S_pairwise_fixed = connect_ports_fixed(S, p, q)
    print("\n--- (C) pairwise DA SUA (engine.connect_ports_fixed) ---")
    print(np.round(S_pairwise_fixed, 6))

    diff_fixed = np.abs(S_pairwise_fixed - S_ground_truth)
    print(f"\nMax abs diff (C vs B, sau khi sua): {diff_fixed.max():.3e}")
    print("=> KHOP SAU KHI SUA" if diff_fixed.max() < 1e-9 else "=> VAN SAI")

    print("\n=== KET LUAN ===")
    print("Loi nam o engine.py dong 21-22 (ham connect_ports): 2 he so (1-S[q,p]) va")
    print("(1-S[p,q]) bi dao cho nhau giua 2 so hang tu cua tu so.")
    print("- Tren mang THUAN NGHICH (S[p,q]=S[q,p], vi du TLINE/RES trong Wilkinson):")
    print("  dao cho khong gay loi vi 2 he so do BANG NHAU -> bug AN HOAN TOAN.")
    print(f"- Tren transistor (S[p,q]={S[p,q]:.4f} != S[q,p]={S[q,p]:.4f}):")
    print(f"  dao cho gay sai lech max_diff = {diff.max():.6f} so voi ground truth (cot B).")
    print(f"  Sau khi dao lai DUNG vi tri (cot C): max_diff = {diff_fixed.max():.2e} -> khop tuyet doi.")
