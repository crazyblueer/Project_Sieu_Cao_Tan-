"""Progress (Update):
Phase 1: z_to_s_1port: phan xa cua 1 cong (Gamma = (Z-Z0)/(Z+Z0)).
Phase 2: z_to_s_Nport: Z->S cho N cong. Hieu: duong cheo = phan xa, ngoai cheo = truyen.
Phase 3: tline_s + beta_l_from_freq: ma tran S cua doan duong truyen, S21 = e^{-j beta*l} (phu thuoc tan so).
         Hieu: day ly tuong ko lam yeu song, chi quay pha; pha quay theo tan so; doan lamda/4 quay 90 do.
Phase 4: cascade: ghep 2 khoi 2 cong noi duoi. Hieu: ghep KHAC nhan ma tran vi co vong doi qua lai;
         mau so 1/(1-A22*B11) = tong vo han cac lan doi. Da chung minh ghep != nhan bang con so.

Phase 5: connect_ports: THUAT TOAN GHEP TONG QUAT - noi cong bat ky voi cong bat ky trong 1 mang.
         Tu duy moi: gom tat ca khoi thanh 1 ma tran lon -> noi dan tung cap cong -> ma tran co lai.
         Khac cascade (chi noi duoi 2 cong): cai nay noi duoc song song, noi ngang, mang nhieu cong.
         Kiem chung: lam lai Phase 4 bang connect_ports -> ra giong het cascade.
         (Day la thuat toan "noi dan tung cap cong" ma de bai yeu cau.)

Phase 6: RAP WILKINSON hoan chinh.
         - Them component: junction3 (noi chu T 3 cong), tline_s_general (day co Zc rieng),
           resistor_series_s (dien tro cach ly).
         - Gom 6 khoi (2 duong 70.7 ohm + 3 noi T + 1 dien tro) -> noi 6 cap cong -> ra S 3x3.
         - KET QUA khop ly thuyet Wilkinson: S11=S22=S33=0, S21=S31=-0.707j, S23=0
           (chia doi cong suat, phoi hop, cach ly).
         - Kiem chung component tline_s_general bang scikit-rf (sau renormalize ve 50 ohm): KHOP 4 chu so.

TODO tiep theo:
- Renormalization (doi tro khang tham chieu) - tu viet, doi chieu scikit-rf.
- Thuat toan thu hai: connection matrix (giai ca mang 1 lan) - de doi chieu voi noi dan tung cap.
- Quet ca dai tan (ko chi 1 diem f0) de ve do thi dap ung.
"""

import numpy as np

#PHASE 1: PHAN XA; Z-> S CHO 1 CONG
def z_to_s_1port(Z, Z0=50.0):
    """" Chuyen tro khang tai Z sang tham so S (he so phan xa) cho mang 1 cong.
    Z, Z0 co the la so phuc. Ho Tro Z la mang (quet theo tan so)"""
    
    Z = np.asarray(Z, dtype = complex)
    Z0 = np.asarray(Z0, dtype = complex)
    
    return (Z - Z0)/(Z + Z0)

#test thu

#Case 1: tai phoi hop hoan hao Z = Z0 = 50 => gamma = 0 
print(z_to_s_1port(50, 50)) # ket qua mong doi: 0

#case 2: ho mach Z = vo cung => gamma = +1 
print(z_to_s_1port(1e12, 50)) # ket qua mong doi: 1

#case 3: ngan mach Z = 0 => gamma = -1
print(z_to_s_1port(0, 50)) # ket qua mong doi: -1

#case 4: tai 100 Ohm tren he 50 Ohm: Z = 100, Z0 = 50 => gamma = (100-50)/(100+50) = 50/150 = 1/3
print(z_to_s_1port(100, 50)) # ket qua mong doi: 0.3333 


#PHASE 2: MOT PORT, CHUYEN Z-> S CHO N CONG
"""bien ma tran 1 cong thanh N cong. 
Y tuong: o one-port, Z va S deu la mot so. O N-port, chung thanh ma tran NXN. Cong thuc tong quat (dang ma tran) cua phep doi Z-> S la:
S = (Z - Z0*I) * inv(Z + Z0*I)
"""

def z_to_s_Nport(Z, Z0=50.0):
    """Chuyen tro khang tai Z sang tham so S (he so phan xa) cho mang N cong.
    Z, Z0 co the la so phuc. Ho Tro Z la ma tran (quet theo tan so)"""
    
    Z = np.asarray(Z, dtype = complex)
    N = Z.shape[0] #so cong 
    I = np.eye(N) 
    S = (Z -Z0*I)  @ np.linalg.inv(Z + Z0*I)
    return S

#TEST THU 
 # Ca A: mang 1x1 phai cho ket qua giong ham one-port cu
print(z_to_s_Nport(np.array([[100]]), 50))

# Ca B: 2x2 cheo (hai tai 100 Ohm doc lap, khong noi nhau)
print(z_to_s_Nport(np.array([[100, 0], [0, 100]]), 50))
# mong doi: [[0.333, 0], [0, 0.333]]  -- cheo la phan xa, ngoai cheo = 0 (khong truyen)

#Ca C: Mach: ong 1 va cong 2 noi nhau qua mot dien tro R = 50ohm. 
Z = np.array([[50, 50], [50, 50]]) # ma tran tro khang cua mang 2 cong
print(z_to_s_Nport(Z, 50))


#Phase 3: involve tan so. O hai phase truoc, ma tran S la mot con so/ma tran co dinh - tai 100Oh thi mai cho gamma=1/3, ko quan tam toi tan so. O phase 3, ta se xem Xung tan so, va ma tran S se thay doi theo tan so.
#vi pha beta*l quay theo f. Day la cho ma "Ket qua phai tinh theo tung tan so"
"""WilkinsoN: gom 2 doan lamda/4; quan trong. Ham nay nhan input: beta_l,, nhung nguoi dung nghi theo tan so va chieu dai day.
Problem: ham tline_s(beta_l) nhan vao goc Beta*l (radian), nhung trong thuc te, cai ban biet la: tan so f, chieu dai day l, va loai day(quyet dinh van toc song)."""

#tinhs ma tran S cho doan duong truyen ly tuong (Z0 day = tro khang tham chieu)
def tline_s(beta_l):
    """Ma tran S doan duong truyen ly tuong (Z0 day = tro khang tham chieu)."""
    s21 = np.exp(-1j * beta_l)
    return np.array([[0,   s21],
                     [s21, 0  ]])

def beta_l_from_freq(f, f0, electrical_length_deg=90):
    """
    Goc dien beta*l cua mot doan duong truyen tai tan so f.
    electrical_length_deg: chieu dai dien cua doan day tai tan so thiet ke f0.
      - 90 do  = doan lambda/4  (mac dinh, vi hay dung nhat)
      - 180 do = doan lambda/2
      - 45 do  = doan lambda/8
    """
    theta0 = np.deg2rad(electrical_length_deg)   # goc tai f0
    return theta0 * (f / f0)                      # ty le theo tan so

#TEST
f0 = 2.4e9   # tan so thiet ke 2.4 GHz

# Tai dung f0: beta_l = 90 deg -> S21 = e^{-j90} = -j
print(tline_s(beta_l_from_freq(2.4e9, f0))[0,1])   # mong doi: ~ -1j  (tuc 0 - 1j)

# Tai 2*f0: beta_l = 180 deg -> S21 = e^{-j180} = -1
print(tline_s(beta_l_from_freq(4.8e9, f0))[0,1])   # mong doi: ~ -1

# Tai f0/2: beta_l = 45 deg -> S21 = e^{-j45} = 0.707 - 0.707j
print(tline_s(beta_l_from_freq(1.2e9, f0))[0,1])   # mong doi: ~ 0.707 - 0.707j


#PHASE 4: GHEP HAI KHOI 2 CONG (noi tiep / cascade)
"""
Noi cong 2 cua khoi A vao cong 1 cua khoi B.
Ghep KHONG phai nhan ma tran: phai giai vong doi qua lai tai cho noi.
Mau so D = 1 - A22*B11 chinh la tong vo han cac lan doi (1 + r + r^2 + ...).
Cach nay con naive, don gian. 
"""

def cascade(A, B):
    """
    Ghep noi tiep hai mang 2 cong A, B (moi cai la ma tran 2x2).
    Noi cong 2 cua A <-> cong 1 cua B.
    Tra ve ma tran S 2x2 cua khoi ghep (cong ngoai = cong 1 cua A, cong 2 cua B).
    """
    A = np.asarray(A, dtype=complex)
    B = np.asarray(B, dtype=complex)

    # Lay rieng tung phan tu cho de doc (dung chi so 0,1 = cong 1,2)
    A11, A12 = A[0,0], A[0,1]
    A21, A22 = A[1,0], A[1,1]
    B11, B12 = B[0,0], B[0,1]
    B21, B22 = B[1,0], B[1,1]

    D = 1 - A22 * B11          # vong phan hoi: 1/(1 - A22*B11) = tong cac lan doi

    S11 = A11 + (A12 * A21 * B11) / D
    S12 = (A12 * B12) / D
    S21 = (A21 * B21) / D
    S22 = B22 + (B12 * B21 * A22) / D

    return np.array([[S11, S12],
                     [S21, S22]])

#Test 1: nối hai đoạn λ/4 → phải ra một đoạn λ/2.
f0 = 2.4e9
theta = beta_l_from_freq(2.4e9, f0)   # = 90 deg tai f0
A = tline_s(theta)   # doan lambda/4
B = tline_s(theta)   # doan lambda/4 thu hai

ghep = cascade(A, B)
print(np.round(ghep, 4))
# mong doi: S21 = e^{-j180} = -1, S11 = 0
# tuc [[0, -1], [-1, 0]]  -- dung la mot doan lambda/2

#Test 2 — ca CÓ phản xạ, để thấy ghép KHÁC nhân:
# Mot khoi co phan xa: dien tro noi tiep R=50 (tu Phase 2, Ca C)
R = z_to_s_Nport(np.array([[50, 50], [50, 50]]), 50)   # [[-1/3, 2/3],[2/3,-1/3]]
line = tline_s(beta_l_from_freq(2.4e9, f0))            # doan lambda/4

ghep_dung = cascade(R, line)         # ghep dung cach
nhan_sai  = R @ line                 # nhan ma tran (CACH SAI)

print("Ghep dung:", np.round(ghep_dung, 4))
print("Nhan sai :", np.round(nhan_sai, 4))
# Hai ket qua se KHAC nhau -> chung minh ghep != nhan khi co phan xa
#--------------------------------------------------------------------------#

#NGUOI C: THUAT TOAN GHEP CONG APPLY tren Wilkinson 

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

# Gom 2 doan lambda/4 thanh 1 mang 4 cong (chua noi)
A = tline_s(beta_l_from_freq(2.4e9, f0))   # cong 0,1
B = tline_s(beta_l_from_freq(2.4e9, f0))   # cong 2,3

big = np.zeros((4,4), dtype=complex)
big[0:2, 0:2] = A      # khoi A o goc tren-trai
big[2:4, 2:4] = B      # khoi B o goc duoi-phai

# Noi cong 1 cua A (index 1) voi cong 1 cua B (index 2)
ket_qua = connect_ports(big, 1, 2)
print(np.round(ket_qua, 4))
# mong doi: [[0, -1], [-1, 0]]  -- GIONG het cascade o Phase 4!


# Duong truyen TONG QUAT: tro khang day Zc, trong he tham chieu Z0 (co the khac nhau)
def tline_s_general(theta, Zc, Z0=50.0):
    """theta = goc dien (rad). Zc = tro khang dac tinh cua day. Z0 = tham chieu."""
    A = np.cos(theta); D = np.cos(theta)
    B = 1j*Zc*np.sin(theta); C = 1j*np.sin(theta)/Zc
    den = A + B/Z0 + C*Z0 + D
    S11 = (A + B/Z0 - C*Z0 - D)/den
    S12 = 2*(A*D - B*C)/den
    S21 = 2/den
    S22 = (-A + B/Z0 - C*Z0 + D)/den
    return np.array([[S11, S12],[S21, S22]])

# Kiem: Zc = Z0 phai thu ve tline_s cu (S11=0, S21=e^{-j theta})
# tai theta=90, Zc=70.71, Z0=50 -> S11 ~ 0.333, S21 ~ -0.943j (day CO phan xa)

#PHASE 6: RAP WILKINSON
# Cho noi T ly tuong 3 cong (node chung): S_ij = 2/3 (i!=j), S_ii = -1/3
def junction3():
    return np.array([[-1/3, 2/3, 2/3],
                     [ 2/3,-1/3, 2/3],
                     [ 2/3, 2/3,-1/3]], dtype=complex)

# Dien tro NOI TIEP (bac cau) R giua 2 cong, he Z0:
def resistor_series_s(R, Z0=50.0):
    S11 = R/(R + 2*Z0); S21 = 2*Z0/(R + 2*Z0)
    return np.array([[S11, S21],[S21, S11]], dtype=complex)


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

f0 = 2.4e9; Z0 = 50.0; theta = np.pi/2; Zc = 50*np.sqrt(2); R = 100.0

jA   = (junction3(),                     ['P1','A1','A2'])     # node vao
L1   = (tline_s_general(theta, Zc, Z0),  ['L1a','L1b'])        # nhanh tren 70.7 ohm
L2   = (tline_s_general(theta, Zc, Z0),  ['L2a','L2b'])        # nhanh duoi 70.7 ohm
jB   = (junction3(),                     ['B1','P2','Br'])     # node ra 2
jC   = (junction3(),                     ['C1','P3','Cr'])     # node ra 3
Rz   = (resistor_series_s(R, Z0),        ['R1','R2'])          # dien tro cach ly

S, lab = block_diag([jA, L1, L2, jB, jC, Rz])
for a,b in [('A1','L1a'),('L1b','B1'),('A2','L2a'),('L2b','C1'),('Br','R1'),('Cr','R2')]:
    S, lab = connect_named(S, lab, a, b)

print("Cong con lai:", lab)          # mong doi: ['P1','P2','P3']
print(np.round(S, 4))

#SO SANH VOI SCIKIT-RF
import skrf as rf
from skrf.media import DefinedGammaZ0

freq = rf.Frequency(2.4, 2.4, 1, 'ghz')
media = DefinedGammaZ0(frequency=freq, z0=50)       # he tham chieu 50 ohm
line = media.line(90, 'deg', z0=50*np.sqrt(2))      # day 70.7 ohm
line.renormalize(50)                                 # ep S ve he 50 ohm  <-- DONG QUAN TRONG

print("Doan day 70.7 ohm trong he 50 ohm — scikit-rf:")
print(np.round(line.s[0], 4))
print("Doan day 70.7 ohm trong he 50 ohm — ham cua minh:")
print(np.round(tline_s_general(np.pi/2, 50*np.sqrt(2), 50), 4))
# Gio hai cai PHAI khop: ca hai deu cho S11 ~ 0.3333, S21 ~ -0.9428j
