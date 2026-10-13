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

TODO tiep theo:
- Thuat toan thu hai: connection matrix (giai ca mang 1 lan) - de doi chieu voi noi dan tung cap.
- Quet ca dai tan (ko chi 1 diem f0) de ve do thi dap ung S theo f.
- Don cau truc file voi nhom (engine.py, solver.py, components.py).
"""
