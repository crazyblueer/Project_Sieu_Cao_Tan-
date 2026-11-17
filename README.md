# Project_Sieu_Cao_Tan-
"""Progress (Update):

Phase 1: z_to_s_1port: phản xạ của 1 cổng (Gamma = (Z-Z0)/(Z+Z0)).

Phase 2: z_to_s_Nport: Z->S cho N cổng. Hiểu: đường chéo = phản xạ, ngoài chéo = truyền.

Phase 3: tline_s + beta_l_from_freq: ma trận S của đoạn đường truyền, S21 = e^{-j beta*l} (phụ thuộc tần số).
         Hiểu: dây lý tưởng không làm yếu sóng, chỉ quay pha; pha quay theo tần số; đoạn lambda/4 quay 90 độ.

Phase 4: cascade: ghép 2 khối 2 cổng nối đuôi. Hiểu: ghép KHÁC nhân ma trận vì có vòng dội qua lại;
         mẫu số 1/(1-A22*B11) = tổng vô hạn các lần dội. Đã chứng minh ghép != nhân bằng con số.

Phase 5: connect_ports: THUẬT TOÁN GHÉP TỔNG QUÁT - nối cổng bất kỳ với cổng bất kỳ trong 1 mạng.
         Tư duy mới: gộp tất cả khối thành 1 ma trận lớn -> nối dần từng cặp cổng -> ma trận co lại.
         Khác cascade (chỉ nối đuôi 2 cổng): cái này nối được song song, nối ngang, mạng nhiều cổng.
         Kiểm chứng: làm lại Phase 4 bằng connect_ports -> ra giống hệt cascade.
         (Đây là thuật toán "nối dần từng cặp cổng" mà đề bài yêu cầu.)

Phase 6: RÁP WILKINSON hoàn chỉnh.
         - Thêm component: junction3 (nối chữ T 3 cổng), tline_s_general (dây có Zc riêng),
           resistor_series_s (điện trở cách ly).
         - Gồm 6 khối (2 đường 70.7 ohm + 3 nối T + 1 điện trở) -> nối 6 cặp cổng -> ra S 3x3.
         - KẾT QUẢ khớp lý thuyết Wilkinson: S11=S22=S33=0, S21=S31=-0.707j, S23=0
           (chia đôi công suất, phối hợp, cách ly).
         - Kiểm chứng component tline_s_general bằng scikit-rf (sau renormalize về 50 ohm): KHỚP 4 chữ số.
Phase 7: VAI A - các hàm biến đổi và kiểm tra ma trận (nền tảng engine).
         - z_to_s_Nport / s_to_z_Nport: đổi qua lại S <-> Z.
         - renormalize: đổi trở kháng tham chiếu (Z0 THỰC), đi vòng qua Z bất biến.
           Kiểm chứng: đoạn dây 70.7 ohm đổi về 50 ohm -> khớp tline_general và scikit-rf.
         - renormalize_general: bản NÂNG CAO, Z0 PHỨC và khác nhau từng cổng (power wave Kurokawa).
           Kiểm chứng: khớp scikit-rf (s_def='power') với Z0 = 50+10j -> đáp ứng yêu cầu đề bài
           "hỗ trợ trở kháng tham chiếu phức và khác nhau ở từng cổng".
         - is_reciprocal / is_lossless / is_passive: kiểm tra tính chất mạng (lưới an toàn debug).
           Hiểu: lossless = không mất năng lượng; passive = không tự sinh năng lượng (điện trở: lossless=False, passive=True).

Phase 8: TÍCH HỢP - nối parser netlist (bản B) với engine ghép (bản+C).
         - solve_network (solver.py): nhận blocks/connections/ext_ports từ netlist,
           tự động gộp khối + nối dần các cặp cổng + giữ lại đúng các cổng ngoài theo thứ tự.
         - Xử lý phân biệt EXT labels (cổng ngoài) với mối nối thật.
         - KẾT QUẢ: chạy 1 netlist Wilkinson văn bản -> ra S khớp lý thuyết (-0.707j, 0, 0).
           Kiểm chứng bằng 3 hàm tính chất: reciprocal=True, passive=True, lossless=False (đúng).
         => Lần đầu project chạy liền mạch: netlist văn bản -> parser -> engine -> ma trận S.

Phase 9: THUẬT TOÁN THỨ HAI - connection matrix (connect_matrix/connect_named_matrix
         trong engine.py), giải cả mạng 1 lần bằng 1 phép nghịch đảo ma trận
         (Redheffer star product tổng quát), thay vì nối dần từng cặp như pairwise.
         Đã gộp vào engine.py (không còn là file riêng connection_matrix.py) vì đây
         cũng là một phần của engine ghép cổng.
         Hàm tích hợp solve_network_global(blocks, connections, ext) (dùng
         connect_named_matrix) nằm trong solver.py, cạnh solve_network() (pairwise)
         - cả hai cùng vai trò "nối netlist với engine", chỉ khác thuật toán nối
         dùng, nên gộp chung 1 file hợp lý hơn để người đọc thấy ngay 2 lựa chọn.
         Kiểm chứng: solve_network_global khớp solve_network (pairwise) tới 1e-16
         trên Wilkinson và mọi netlist khác (xem compare_algorithms.py).

Phase 10: PHÁT HIỆN & SỬA BUG - connect_ports() (bản GỐC) bị đảo 2 hệ số (1-S[p,q])
         <-> (1-S[q,p]) trong công thức nối dần từng cặp. Bug ẨN HOÀN TOÀN trên mạng
         THUẬN NGHỊCH (TLINE/RES/Wilkinson, S[p,q]=S[q,p]) nhưng LỘ RÕ trên mạng
         KHÔNG thuận nghịch (transistor, S12 != S21) (file demo cũ transistor_bug_demo.py đã được xóa).
         Đã sửa thành connect_ports_fixed() (bản CHÍNH THỨC, dùng qua connect_named()),
         GIỮ LẠI connect_ports() bản gốc trong engine.py để đối chiếu/minh họa lỗi.

Phase 11: MONTE-CARLO DUNG SAI LINH KIỆN (monte_carlo.py).
         - Nhiễu ngẫu nhiên giá trị R/L/C của một netlist theo PHÂN BỐ GAUSS
           (sigma = tol_frac / 3, quy ước dung sai công bố = 3*sigma), giải
           lại mạng bằng solve_network_global() (connection-matrix) tại các
           tần số cần khảo sát, lặp lại N lần (trial).
         - Thống kê |S11|/|S21| (dB) trên toàn bộ trial: mean/std/percentile
           5-50-95% + tỷ lệ ĐẠT spec (yield) -> đánh giá dung sai linh kiện
           ảnh hưởng thế nào đến hiệu năng mạch thực tế (không chỉ giá trị
           lý tưởng tại điểm danh định).
         - Đã chạy trên filter_lc.net (nhiễu C/L +-5%) và amplifier.net
           (nhiễu điện trở hồi tiếp RFB +-10%), xuất histogram PNG cho mỗi
           cặp tần số/chỉ tiêu.

Phase 12: QUÉT TOÀN DẢI TẦN & VẼ ĐỒ THỊ S(f) (compare_algorithms.py).
         - plot_sweep() quét nhiều điểm tần số (không chỉ 1 điểm f0), vẽ biên độ
           |S_ij| (dB) và pha S_ij theo tần số cho đủ cả 3 cách (pairwise/
           connection-matrix/scikit-rf) trên mỗi netlist, chạy qua
           run_case(max_points=None).
         - KẾT QUẢ: xuất ảnh sweep_<ten>.png cho từng netlist (wilkinson,
           branchline, filter_lc, filter_stub, amplifier); sai lệch pairwise
           vs connection-matrix dưới 1e-16, so với scikit-rf dưới 1e-6 trên
           toàn dải tần (đạt yêu cầu độ chính xác của đề bài).
         - Chạy được với netlist bất kỳ, không cần sửa code:
           python compare_algorithms.py <ten_file.net> [file2.net ...]

Phase 13: TRỞ KHÁNG THAM CHIẾU Z0 PHỨC, KHÁC NHAU TỪNG CỔNG + HỢP NHẤT MÃ NGUỒN (đề tài 7*).
         Tóm tắt thay đổi so với source code gốc (Phase 1-12):
         - Mục tiêu: cổng ngoài (và mọi chân) có thể có Z0 THỰC hoặc PHỨC, KHÁC NHAU
           từng cổng, theo sóng công suất (Kurokawa) - khớp scikit-rf s_def='power'.
           Không khai Z0 -> 50 ohm, kết quả giống bản gốc.
         - Cách dùng Z0:
             + trong netlist:   PORT 2 OUT Z0=50+10j   (thực hoặc phức)
             + lúc chạy:        net.set_port_z0({1: 75, 2: "50+10j"})   (không cần sửa .net)
             + file .s2p:       ts.s_interp(f, z0=[50, 75])  (chuẩn hóa R trong header về Z0 bất kỳ)
             + Monte-Carlo:     run_monte_carlo(..., port_z0={1: 75, 2: "50+10j"})
           Ưu tiên: set_port_z0 > PORT ... Z0= trong .net > mặc định 50.
         - Thay đổi theo file:
             + engine.py (viết lại): thêm gamma_c (khối nối power-wave giữa 2 chân có Z0
               khác nhau); connect_ports(S, p, q, Zp, Zq) là bản ĐÚNG; connect_matrix(S,
               pairs, z0); một hàm connect(S, labels, name_pairs, z0_ports,
               method="matrix"|"pairwise"). Gỡ: connect_ports bản lỗi (Phase 10),
               connect_ports_fixed, connect_named, connect_named_matrix.
             + solver.py (viết lại): solve_network(blocks, connections, ext, pin_z0,
               ext_z0, z0, method="matrix") thay cho solve_network và
               solve_network_global; thêm solve_netlist_at_freq(net, f, method=).
               Mặc định đổi từ pairwise sang connection-matrix.
             + netlist.py: cú pháp PORT n nút [Z0=...]; build_network(f, z0) trả
               Network(blocks, connections, ext_labels, pin_z0, ext_z0) thay cho
               build_network_at_freq; thêm set_port_z0().
             + touchstone.py: s_interp(f, mode, z0=None); cảnh báo "R != 50" dời từ
               load_s2p sang s_interp (chỉ khi không truyền z0, một lần mỗi file).
             + monte_carlo.py: tham số port_z0, khóa kết quả z0_ports, 2 demo Z0 phức
               (monte_carlo_filter_lc_z0c.png, monte_carlo_amplifier_z0c.png). Spec có thể
               khai tương đối quanh danh định: spec={'S21': ('nom', 0.15)} = danh định ± 0.15 dB
               (tính riêng từng tần số) Spec demo amplifier S21 đổi từ (10, 30) dB sang danh định ± 0.15 dB.
             + compare_algorithms.py: dùng API mới; thêm bảng Z0 phức khác nhau từng
               cổng so với scikit-rf; bỏ phần chứng minh bug cũ; thêm mục "SAI KHAC TREN
               LUOI 1001 DIEM" (max của |A-B|, |A-C|, |B-C| trên 1001 điểm, cả Z0 = 50
               lẫn Z0 phức).
             + Toàn bộ comment/docstring trong code (và phần chú thích trong các file
               .net, .s2p) được viết lại bằng tiếng Việt có dấu; chuỗi in ra (print) giữ
               tiếng Việt không dấu. Logic code không đổi (đã so sánh AST).
             + main.py: 2 dòng gọi API mới (kết quả in ra y hệt bản gốc).
             + Không đổi logic: components.py, các file .net, fet.s2p, test.s2p (chỉ đổi
               phần chú thích sang tiếng Việt có dấu) và các ảnh sweep_*.png.
             + Không đổi: trial_error.py (file draft, giữ nguyên bản gốc).
             + Đã xóa: transistor_bug_demo.py (bản lỗi đã gỡ khỏi engine.py).
         - Tự kiểm thử: mỗi module có self_test() (python engine.py | solver.py |
           netlist.py | touchstone.py | monte_carlo.py --test).
         - Kiểm chứng so với bản gốc ở Z0 mặc định: connection-matrix giống hệt từng bit;
           pairwise lệch <= 2e-15; Z0 phức khác nhau từng cổng khớp scikit-rf tới ~1e-14.
         - Lưu ý tương thích: tên hàm cũ (connect_named*, solve_network_global,
           build_network_at_freq) không còn; dùng
               S = solve_network(*net.build_network(f))                    # matrix
               S = solve_network(*net.build_network(f), method="pairwise")
               S = solve_netlist_at_freq(net, f)                           # một dòng

CẤU TRÚC FILE HIỆN TẠI (đã đơn giản hóa theo nhóm):
  - engine.py        : nền tảng ghép cổng - gamma_c, connect_ports (1 cặp), connect_matrix
                        (giải 1 lần), connect(..., method="matrix"|"pairwise").
  - components.py    : ma trận S các phần tử vi sóng cơ bản (R/L/C, TLINE, junction3,...),
                        renormalize / renormalize_general.
  - netlist.py        : đọc netlist văn bản (kể cả PORT ... Z0=), gọi components.py/
                        touchstone.py, xuất Network qua build_network(); set_port_z0().
  - touchstone.py     : đọc/ghi/nội suy file Touchstone .s2p, chuẩn hóa về Z0 bất kỳ.
  - solver.py         : tích hợp netlist + engine - solve_network(..., method=) và
                        solve_netlist_at_freq(net, f, method=).
  - compare_algorithms.py : kiểm chứng chéo 3 cách (pairwise/connection-matrix/scikit-rf)
                        trên mỗi netlist, cả Z0 mặc định lẫn Z0 phức.
  - main.py / trial_error.py : kiểm thử/draft ban đầu (z_to_s, renormalize, Touchstone).
  - monte_carlo.py    : mô phỏng Monte-Carlo dung sai linh kiện (phân bố GAUSS),
                        thống kê + vẽ histogram S11/S21 (dB), tính yield theo spec.
  - *.net             : các file netlist ví dụ (wilkinson, branchline, filter_lc,
                        filter_stub, amplifier).
TRẠNG THÁI DỰ ÁN (HIỆN TẠI DỪNG Ở ĐÂY):
- Quét cả dải tần (không chỉ 1 điểm f0) để vẽ đồ thị đáp ứng S theo f cho
  đủ cả 3 cách (pairwise/connection-matrix/scikit-rf) - ĐÃ XONG, xem
  compare_algorithms.py (hàm plot_sweep(), chạy qua run_case(max_points=None)).
- Phase 12 đã hoàn tất; Phase 13 (Z0 phức + hợp nhất mã nguồn) là bước cập nhật
  mới nhất. Monte-Carlo dung sai
  linh kiện (Phase 11, monte_carlo.py) KHÔNG bị ảnh hưởng, vẫn thuộc tiến
  độ chính thức.

CẬP NHẬT GẦN ĐÂY (đồ thị quét tần số + netlist tuỳ ý):
- plot_sweep() (compare_algorithms.py) nay vẽ PHA S_ij cho đủ cả 3 cách
  (pairwise/conn-matrix/scikit-rf), giống panel biên độ (trước đây panel
  pha chỉ có 1 đường pairwise). Bỏ dòng chú giải thừa trong tiêu đề panel
  sai số pairwise-vs-conn-matrix.
- CHẠY VỚI NETLIST TUỲ Ý (tự động sinh ảnh cho netlist bất kỳ, KHÔNG cần
  sửa code): python compare_algorithms.py <ten_file.net> [file2.net ...]
  run_case()/plot_sweep() không hard-code tên hay số cổng, nên MỌI file
  .net hợp cú pháp netlist.py (FREQ/PORT + RES/CAP/IND/TLINE/MLIN/CLIN/
  STUB/OPEN/SHORT/S2P) sẽ tự động được: kiểm chứng chéo 3 cách + vẽ PNG
  sweep_<ten>.png. check_theory() (đối chiếu lý thuyết) chỉ biết 5 mạch
  ví dụ có sẵn và sẽ tự bỏ qua (không lỗi) cho netlist mới; bước vẽ đồ
  thị và kiểm chứng số vẫn chạy đầy đủ. Không truyền đối số -> vẫn chạy
  5 netlist ví dụ như trước.
"""
