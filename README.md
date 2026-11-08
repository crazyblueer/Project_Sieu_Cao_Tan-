# Project_Sieu_Cao_Tan-
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
Phase 7: VAI A - cac ham bien doi va kiem tra ma tran (nen tang engine).
         - z_to_s_Nport / s_to_z_Nport: doi qua lai S <-> Z.
         - renormalize: doi tro khang tham chieu (Z0 THUC), di vong qua Z bat bien.
           Kiem chung: doan day 70.7 ohm doi ve 50 ohm -> khop tline_general va scikit-rf.
         - renormalize_general: ban NANG CAO, Z0 PHUC va khac nhau tung cong (power wave Kurokawa).
           Kiem chung: khop scikit-rf (s_def='power') voi Z0 = 50+10j -> dap ung yeu cau de bai
           "ho tro tro khang tham chieu phuc va khac nhau o tung cong".
         - is_reciprocal / is_lossless / is_passive: kiem tra tinh chat mang (luoi an toan debug).
           Hieu: lossless = ko mat nang luong; passive = ko tu sinh nang luong (dien tro: lossless=False, passive=True).

Phase 8: TICH HOP - noi parser netlist (ban B) voi engine ghep (ban+C).
         - solve_network (solver.py): nhan blocks/connections/ext_ports tu netlist,
           tu dong gom khoi + noi dan cac cap cong + giu lai dung cac cong ngoai theo thu tu.
         - Xu ly phan biet EXT labels (cong ngoai) voi moi noi that.
         - KET QUA: chay 1 netlist Wilkinson van ban -> ra S khop ly thuyet (-0.707j, 0, 0).
           Kiem chung bang 3 ham tinh chat: reciprocal=True, passive=True, lossless=False (dung).
         => Lan dau project chay lien mach: netlist van ban -> parser -> engine -> ma tran S.

Phase 9: THUAT TOAN THU HAI - connection matrix (connect_matrix/connect_named_matrix
         trong engine.py), giai ca mang 1 lan bang 1 phep nghich dao ma tran
         (Redheffer star product tong quat), thay vi noi dan tung cap nhu pairwise.
         Da gop vao engine.py (khong con la file rieng connection_matrix.py) vi day
         cung la mot phan cua engine ghep cong.
         Ham tich hop solve_network_global(blocks, connections, ext) (dung
         connect_named_matrix) nam trong solver.py, canh solve_network() (pairwise)
         - ca hai cung vai tro "noi netlist voi engine", chi khac thuat toan noi
         dung, nen gop chung 1 file hop ly hon de nguoi doc thay ngay 2 lua chon.
         Kiem chung: solve_network_global khop solve_network (pairwise) toi 1e-16
         tren Wilkinson va moi netlist khac (xem compare_algorithms.py).

Phase 10: PHAT HIEN & SUA BUG - connect_ports() (ban GOC) bi dao 2 he so (1-S[p,q])
         <-> (1-S[q,p]) trong cong thuc noi dan tung cap. Bug AN HOAN TOAN tren mang
         THUAN NGHICH (TLINE/RES/Wilkinson, S[p,q]=S[q,p]) nhung LO RO tren mang
         KHONG thuan nghich (transistor, S12 != S21) - xem transistor_bug_demo.py.
         Da sua thanh connect_ports_fixed() (ban CHINH THUC, dung qua connect_named()),
         GIU LAI connect_ports() ban goc trong engine.py de doi chieu/minh hoa loi.

Phase 11: MONTE-CARLO DUNG SAI LINH KIEN (monte_carlo.py).
         - Nhieu ngau nhien gia tri R/L/C cua mot netlist theo PHAN BO GAUSS
           (sigma = tol_frac / 3, quy uoc dung sai cong bo = 3*sigma), giai
           lai mang bang solve_network_global() (connection-matrix) tai cac
           tan so can khao sat, lap lai N lan (trial).
         - Thong ke |S11|/|S21| (dB) tren toan bo trial: mean/std/percentile
           5-50-95% + ty le DAT spec (yield) -> danh gia dung sai linh kien
           anh huong the nao den hieu nang mach thuc te (khong chi gia tri
           ly tuong tai diem danh dinh).
         - Da chay tren filter_lc.net (nhieu C/L +-5%) va amplifier.net
           (nhieu dien tro hoi tiep RFB +-10%), xuat histogram PNG cho moi
           cap tan so/chi tieu.

CAU TRUC FILE HIEN TAI (da don gon theo nhom):
  - engine.py        : nen tang ghep cong - connect_ports (goc, co bug, GIU lam
                        tai lieu), connect_ports_fixed/connect_named (ban
                        CHINH THUC da sua, pairwise), connect_matrix/
                        connect_named_matrix (connection-matrix, giai 1 lan).
  - components.py    : ma tran S cac phan tu vi song co ban (R/L/C, TLINE, junction3,...).
  - netlist.py        : doc netlist van ban, goi components.py/touchstone.py, xuat
                        (blocks, connections, ext_ports).
  - touchstone.py     : doc/ghi/noi suy file Touchstone .s2p.
  - solver.py         : tich hop netlist + engine - solve_network() (pairwise,
                        dung connect_named) va solve_network_global() (connection-
                        matrix, dung connect_named_matrix) - ca 2 cung interface,
                        de so sanh truc tiep (xem compare_algorithms.py).
  - compare_algorithms.py : kiem chung cheo 3 cach (pairwise/connection-matrix/scikit-rf)
                        tren moi netlist, va demo sai lech cua ban loi cu.
  - transistor_bug_demo.py : demo doc lap minh hoa bug connect_ports() tren mang
                        KHONG thuan nghich (transistor), goi thang connect_ports/
                        connect_ports_fixed/connect_matrix tu engine.py.
  - main.py / trial_error.py : kiem thu/draft ban dau (z_to_s, renormalize, Touchstone).
  - monte_carlo.py    : mo phong Monte-Carlo dung sai linh kien (phan bo GAUSS),
                        thong ke + ve histogram S11/S21 (dB), tinh yield theo spec.
  - *.net             : cac file netlist vi du (wilkinson, branchline, filter_lc,
                        filter_stub, amplifier).

TODO tiep theo:
- Quet ca dai tan (ko chi 1 diem f0) de ve do thi dap ung S theo f (da co trong
  compare_algorithms.py qua net.freqs, co the mo rong them cho main.py).
"""
