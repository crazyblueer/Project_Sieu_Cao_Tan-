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
         KHÔNG thuận nghịch (transistor, S12 != S21) - xem transistor_bug_demo.py.
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

CẤU TRÚC FILE HIỆN TẠI (đã đơn giản hóa theo nhóm):
  - engine.py        : nền tảng ghép cổng - connect_ports (gốc, có bug, GIỮ làm
                        tài liệu), connect_ports_fixed/connect_named (bản
                        CHÍNH THỨC đã sửa, pairwise), connect_matrix/
                        connect_named_matrix (connection-matrix, giải 1 lần).
  - components.py    : ma trận S các phần tử vi sóng cơ bản (R/L/C, TLINE, junction3,...).
  - netlist.py        : đọc netlist văn bản, gọi components.py/touchstone.py, xuất
                        (blocks, connections, ext_ports).
  - touchstone.py     : đọc/ghi/nội suy file Touchstone .s2p.
  - solver.py         : tích hợp netlist + engine - solve_network() (pairwise,
                        dùng connect_named) và solve_network_global() (connection-
                        matrix, dùng connect_named_matrix) - cả 2 cùng interface,
                        dễ so sánh trực tiếp (xem compare_algorithms.py).
  - compare_algorithms.py : kiểm chứng chéo 3 cách (pairwise/connection-matrix/scikit-rf)
                        trên mỗi netlist, và demo sai lệch của bản lỗi cũ.
  - transistor_bug_demo.py : demo độc lập minh họa bug connect_ports() trên mạng
                        KHÔNG thuận nghịch (transistor), gọi thẳng connect_ports/
                        connect_ports_fixed/connect_matrix từ engine.py.
  - main.py / trial_error.py : kiểm thử/draft ban đầu (z_to_s, renormalize, Touchstone).
  - monte_carlo.py    : mô phỏng Monte-Carlo dung sai linh kiện (phân bố GAUSS),
                        thống kê + vẽ histogram S11/S21 (dB), tính yield theo spec.
  - *.net             : các file netlist ví dụ (wilkinson, branchline, filter_lc,
                        filter_stub, amplifier).

TODO tiếp theo:
- Quét cả dải tần (không chỉ 1 điểm f0) để vẽ đồ thị đáp ứng S theo f (đã có trong
  compare_algorithms.py qua net.freqs, có thể mở rộng thêm cho main.py).
"""
