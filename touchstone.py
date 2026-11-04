"""
Module: touchstone.py
Chức năng:
1. Đọc file Touchstone v1 2 cổng (.s2p) từ máy đo VNA / phần mềm mô phỏng.
2. Chuyển dữ liệu sang mảng ma trận S phức (num_freqs, 2, 2).
3. Nội suy ma trận S tại tần số bất kỳ trong dải dữ liệu (cấm ngoại suy).
4. Ghi ma trận S ra file .s2p (để xuất kết quả mô phỏng).
"""

import warnings
import numpy as np

FREQ_MULT = {'HZ': 1.0, 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9}


class Touchstone2P:
    """Lưu trữ và xử lý dữ liệu file .s2p của mạng 2 cổng."""

    def __init__(self, filename=None):
        self.filename = filename
        self.freqs = np.array([])        # Tần số (Hz), tăng dần
        self.s_matrices = np.array([])   # (num_freqs, 2, 2), complex
        self.z0 = 50.0                   # Trở kháng tham chiếu của file (Ohm)
        self.freq_unit = "GHZ"           # Mặc định của chuẩn Touchstone
        self.data_format = "MA"          # Mặc định của chuẩn Touchstone

        if filename is not None:
            self.load(filename)

    # -----------------------------------------------------------------
    def load(self, filename):
        """
        Đọc và phân tích cú pháp file .s2p.
        - Chú thích: mọi thứ sau '!' (kể cả cuối dòng dữ liệu) bị bỏ qua.
        - Dòng tùy chọn '#': # <Hz|kHz|MHz|GHz> S <MA|DB|RI> R <Z0>
        - Dữ liệu: mỗi bản ghi 9 số (f, S11, S21, S12, S22), được phép xuống dòng.
        """
        self.filename = filename
        mult = FREQ_MULT[self.freq_unit]
        fmt = self.data_format
        option_seen = False
        numbers = []

        with open(filename, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                line_str = line.split('!', 1)[0].strip()      # bỏ chú thích
                if not line_str:
                    continue

                if line_str.startswith('#'):
                    if option_seen:                           # chuẩn: chỉ dòng '#' đầu tiên có hiệu lực
                        continue
                    option_seen = True
                    tokens = line_str[1:].upper().split()
                    i = 0
                    while i < len(tokens):
                        tok = tokens[i]
                        if tok in FREQ_MULT:
                            mult = FREQ_MULT[tok]
                            self.freq_unit = tok
                        elif tok in ("MA", "DB", "RI"):
                            fmt = tok
                            self.data_format = tok
                        elif tok in ("Y", "Z", "H", "G"):
                            raise NotImplementedError(f"Chỉ hỗ trợ tham số S, file dùng '{tok}'")
                        elif tok == "R" and i + 1 < len(tokens):
                            self.z0 = float(tokens[i + 1])
                            i += 1
                        i += 1
                    continue

                try:
                    numbers.extend(float(x) for x in line_str.split())
                except ValueError:
                    raise ValueError(f"{filename}, dòng {line_no}: dữ liệu không phải số: {line_str!r}")

        if len(numbers) % 9 != 0:
            raise ValueError(f"{filename}: số lượng giá trị ({len(numbers)}) không chia hết cho 9 "
                             f"-> file không phải 2 cổng hoặc bị thiếu dữ liệu")
        data = np.array(numbers).reshape(-1, 9)
        if data.shape[0] == 0:
            raise ValueError(f"{filename}: không có dòng dữ liệu")

        freq_list, s_list = [], []
        for row in data:
            s11 = self._parse_s_pair(row[1], row[2], fmt)
            s21 = self._parse_s_pair(row[3], row[4], fmt)
            s12 = self._parse_s_pair(row[5], row[6], fmt)
            s22 = self._parse_s_pair(row[7], row[8], fmt)
            # Thứ tự trong file 2 cổng là S11, S21, S12, S22 -> lắp đúng vị trí ma trận
            s_list.append(np.array([[s11, s12],
                                    [s21, s22]], dtype=complex))
            freq_list.append(row[0] * mult)

        self.freqs = np.array(freq_list, dtype=float)
        self.s_matrices = np.array(s_list, dtype=complex)

        if np.any(np.diff(self.freqs) <= 0):
            raise ValueError(f"{filename}: tần số phải tăng dần và không trùng lặp")
        if self.z0 != 50.0:
            warnings.warn(f"{filename}: trở kháng tham chiếu R = {self.z0} Ohm khác 50 Ohm; "
                          f"cần chuẩn hóa lại (renormalize) trước khi ghép vào mạch 50 Ohm.")
        return self

    # -----------------------------------------------------------------
    @staticmethod
    def _parse_s_pair(val1, val2, fmt):
        """
        Quy đổi cặp (val1, val2) sang số phức theo định dạng fmt.
        - MA: |S|, góc (độ)    - DB: 20*log10|S|, góc (độ)    - RI: phần thực, phần ảo
        """
        if fmt == "MA":
            return val1 * np.exp(1j * np.deg2rad(val2))
        if fmt == "DB":
            return 10.0 ** (val1 / 20.0) * np.exp(1j * np.deg2rad(val2))
        if fmt == "RI":
            return complex(val1, val2)
        raise ValueError(f"Định dạng {fmt} không hợp lệ.")

    # -----------------------------------------------------------------
    def s_interp(self, f_target, mode="ri"):
        """
        Nội suy tuyến tính ma trận S tại tần số f_target (Hz).
        mode = "ri": nội suy phần thực, phần ảo độc lập (mặc định, giống scikit-rf).
        mode = "ma": nội suy biên độ và pha đã unwrap (tốt hơn khi pha quay nhanh).
        Báo lỗi nếu f_target nằm ngoài dải dữ liệu (np.interp sẽ lặng lẽ lấy giá trị biên).
        """
        if len(self.freqs) == 0:
            raise ValueError("Chưa có dữ liệu tần số để nội suy.")
        f_target = float(f_target)
        f_lo, f_hi = self.freqs[0], self.freqs[-1]
        tol = 1e-9 * f_hi
        if f_target < f_lo - tol or f_target > f_hi + tol:
            raise ValueError(f"Tần số {f_target/1e9:.4f} GHz nằm ngoài dải dữ liệu "
                             f"[{f_lo/1e9:.4f}, {f_hi/1e9:.4f}] GHz của {self.filename}")

        s_target = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                sij = self.s_matrices[:, i, j]
                if mode == "ri":
                    re = np.interp(f_target, self.freqs, sij.real)
                    im = np.interp(f_target, self.freqs, sij.imag)
                    s_target[i, j] = re + 1j * im
                elif mode == "ma":
                    mag = np.interp(f_target, self.freqs, np.abs(sij))
                    ph = np.interp(f_target, self.freqs, np.unwrap(np.angle(sij)))
                    s_target[i, j] = mag * np.exp(1j * ph)
                else:
                    raise ValueError("mode phải là 'ri' hoặc 'ma'")
        return s_target


def load_s2p(filename):
    """
    Hàm tiện ích để Người C import và dùng ngay.
        ts = load_s2p("fet.s2p");  S = ts.s_interp(2.4e9)
    """
    return Touchstone2P(filename)


def save_s2p(filename, freqs, s_matrices, z0=50.0, fmt="RI", freq_unit="GHz", comment=""):
    """Ghi mảng S (num_freqs, 2, 2) ra file .s2p (thứ tự S11 S21 S12 S22)."""
    fmt = fmt.upper()
    mult = FREQ_MULT[freq_unit.upper()]
    with open(filename, "w", encoding="utf-8") as fh:
        for line in comment.splitlines():
            fh.write(f"! {line}\n")
        fh.write(f"# {freq_unit} S {fmt} R {z0:g}\n")
        for f, S in zip(freqs, s_matrices):
            vals = []
            for s in (S[0, 0], S[1, 0], S[0, 1], S[1, 1]):
                if fmt == "RI":
                    vals += [s.real, s.imag]
                elif fmt == "MA":
                    vals += [abs(s), np.angle(s, deg=True)]
                else:
                    vals += [20 * np.log10(abs(s)), np.angle(s, deg=True)]
            fh.write(f"{f / mult:.10g} " + " ".join(f"{v:.12g}" for v in vals) + "\n")


# =====================================================================
# BỘ TEST TỰ ĐỘNG & TẠO FILE MẪU DEMO
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("KIỂM THỬ MODULE TOUCHSTONE.PY (NGƯỜI B)")
    print("=" * 60)

    # 1. File mẫu: đoạn dây lambda/4 tại 2.4 GHz (có chú thích cuối dòng, bản ghi xuống dòng)
    dummy = "sample_line.s2p"
    with open(dummy, "w", encoding="utf-8") as f:
        f.write("""! File mau: doan day ly tuong lambda/4 tai 2.4 GHz
# GHz S MA R 50.0
2.0  0.00 0.00  1.00 -75.00  1.00 -75.00  0.00 0.00   ! chu thich cuoi dong
2.4  0.00 0.00  1.00 -90.00
     1.00 -90.00  0.00 0.00                            ! ban ghi xuong dong
3.0  0.00 0.00  1.00 -112.5  1.00 -112.5  0.00 0.00
""")
    ts = load_s2p(dummy)
    assert len(ts.freqs) == 3 and np.isclose(ts.freqs[1], 2.4e9), "Lỗi đọc tần số"
    assert np.isclose(ts.s_matrices[1, 1, 0], -1j), "Lỗi parse S21 tại 2.4 GHz"
    print("[PASS] 1. Đọc file (chú thích cuối dòng, bản ghi xuống dòng), S21(2.4 GHz) = -j")

    # 2. Ba định dạng cho cùng dữ liệu
    for fmt in ("RI", "MA", "DB"):
        save_s2p("tmp.s2p", ts.freqs, ts.s_matrices + 0.1, fmt=fmt)
        assert np.allclose(load_s2p("tmp.s2p").s_matrices, ts.s_matrices + 0.1, atol=1e-10), f"Lỗi {fmt}"
    print("[PASS] 2. Ghi rồi đọc lại đúng ở cả ba định dạng RI, MA, DB")

    # 3. Nội suy tại trung điểm 2.2 GHz
    s22 = ts.s_interp(2.2e9)
    s22ma = ts.s_interp(2.2e9, mode="ma")
    assert np.isclose(abs(s22[1, 0]), np.cos(np.deg2rad(7.5))), "Lỗi nội suy RI"
    assert np.isclose(abs(s22ma[1, 0]), 1.0) and np.isclose(np.angle(s22ma[1, 0], deg=True), -82.5), "Lỗi nội suy MA"
    print(f"[PASS] 3. Nội suy 2.2 GHz: RI |S21| = {abs(s22[1,0]):.4f} (= cos 7.5°), "
          f"MA |S21| = {abs(s22ma[1,0]):.4f}, góc {np.angle(s22ma[1,0], deg=True):.1f}°")

    # 4. Cấm ngoại suy
    try:
        ts.s_interp(3.5e9)
        raise AssertionError("Không báo lỗi ngoại suy")
    except ValueError:
        print("[PASS] 4. Báo lỗi khi tần số nằm ngoài dải dữ liệu")

    import os
    os.remove("tmp.s2p")
    print("\n=> TẤT CẢ KIỂM THỬ ĐẠT.")
