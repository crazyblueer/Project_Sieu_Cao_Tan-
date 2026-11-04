"""
Module: components.py
Cung cấp ma trận tán xạ S (Scattering Matrix) cho các phần tử vi sóng cơ bản
trong hệ trở kháng tham chiếu thực Z0 (mặc định 50 Ohm).
Mỗi hàm tính tại MỘT tần số f (Hz) và trả về mảng NumPy complex kích thước (N, N).

Danh mục:
  0. to_si                         - chuẩn hóa tiền tố đơn vị
  1-3. R, L, C nối tiếp / song song (mạng 2 cổng)
  4. load_1port                    - trở kháng Z nối đất (mạng 1 cổng)
  5. Đường truyền: tline_ideal, tline_general, beta_l_from_freq
  6. Vi dải: microstrip_z0_eeff, microstrip
  7. Đường ghép đối xứng: coupled_line (4 cổng)
  8. Hở mạch, ngắn mạch, stub; mối nối chữ T và mối nối N cổng
"""

import re
import numpy as np
from scipy.constants import c as C0, mu_0

ETA0 = mu_0 * C0          # Trở kháng sóng chân không ~ 376.73 Ohm


# =====================================================================
# 0. CHUẨN HÓA TIỀN TỐ ĐƠN VỊ
# =====================================================================

_PREFIX = {'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'µ': 1e-6,
           'm': 1e-3, 'k': 1e3, 'g': 1e9, 't': 1e12}
_UNITS = ('hz', 'ohm', 'h', 'f', 's', 'm', 'deg')
_NUM_RE = re.compile(r'^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Zµ]*)\s*$')


def to_si(value):
    """
    Chuyển giá trị có tiền tố SI về số thực theo đơn vị cơ bản.
    Quy ước SPICE (không phân biệt hoa/thường):
        '10n' -> 1e-8   '1pF' -> 1e-12   '2.4GHz' -> 2.4e9   '4.7uH' -> 4.7e-6
        '1Meg' -> 1e6   '100MHz' -> 1e8  '5mm' -> 5e-3       '70.71' -> 70.71
    LƯU Ý: 'm' / 'M' đứng một mình = milli. Mega phải viết 'Meg' hoặc 'MHz'.
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    m = _NUM_RE.match(str(value))
    if not m:
        raise ValueError(f"Giá trị không hợp lệ: {value!r}")
    num, suffix = float(m.group(1)), m.group(2).lower()
    if suffix == '':
        return num
    if suffix.startswith('meg'):
        return num * 1e6
    if suffix == 'mhz':
        return num * 1e6
    if suffix in _PREFIX:                      # một ký tự = tiền tố
        return num * _PREFIX[suffix]
    if suffix in _UNITS:                       # chỉ có đơn vị
        return num
    p, rest = suffix[0], suffix[1:]
    if p in _PREFIX and rest in _UNITS:        # tiền tố + đơn vị
        return num * _PREFIX[p]
    raise ValueError(f"Tiền tố/đơn vị không nhận dạng được: {value!r}")


# =====================================================================
# Hai khối cơ sở: trở kháng nối tiếp và dẫn nạp song song
# =====================================================================

def _series_z(Z, Z0=50.0):
    """S11 = S22 = Z/(Z+2Z0),  S21 = S12 = 2Z0/(Z+2Z0)."""
    den = Z + 2.0 * Z0
    s11 = Z / den
    s21 = 2.0 * Z0 / den
    return np.array([[s11, s21],
                     [s21, s11]], dtype=complex)


def _shunt_y(Y, Z0=50.0):
    """S11 = S22 = -Y*Z0/(Y*Z0+2),  S21 = S12 = 2/(Y*Z0+2)."""
    den = Y * Z0 + 2.0
    s11 = -(Y * Z0) / den
    s21 = 2.0 / den
    return np.array([[s11, s21],
                     [s21, s11]], dtype=complex)


# =====================================================================
# 1. ĐIỆN TRỞ - MẠNG 2 CỔNG
# =====================================================================

def resistor_series(R, Z0=50.0):
    """Điện trở R mắc NỐI TIẾP: S11 = R/(R+2Z0), S21 = 2Z0/(R+2Z0)."""
    return _series_z(complex(float(R)), float(Z0))


def resistor_shunt(R, Z0=50.0):
    """Điện trở R mắc SONG SONG xuống GND: S11 = -Z0/(Z0+2R), S21 = 2R/(Z0+2R)."""
    return _shunt_y(1.0 / float(R), float(Z0))


# =====================================================================
# 2. TỤ ĐIỆN - MẠNG 2 CỔNG
# =====================================================================

def capacitor_series(C, f, Z0=50.0):
    """Tụ C mắc NỐI TIẾP: Z_C = 1/(j*2*pi*f*C)."""
    Z_C = 1.0 / (1j * 2.0 * np.pi * float(f) * float(C))
    return _series_z(Z_C, float(Z0))


def capacitor_shunt(C, f, Z0=50.0):
    """Tụ C mắc SONG SONG xuống GND: Y_C = j*2*pi*f*C."""
    Y_C = 1j * 2.0 * np.pi * float(f) * float(C)
    return _shunt_y(Y_C, float(Z0))


# =====================================================================
# 3. CUỘN CẢM - MẠNG 2 CỔNG
# =====================================================================

def inductor_series(L, f, Z0=50.0):
    """Cuộn cảm L mắc NỐI TIẾP: Z_L = j*2*pi*f*L."""
    Z_L = 1j * 2.0 * np.pi * float(f) * float(L)
    return _series_z(Z_L, float(Z0))


def inductor_shunt(L, f, Z0=50.0):
    """Cuộn cảm L mắc SONG SONG xuống GND: Y_L = 1/(j*2*pi*f*L)."""
    Y_L = 1.0 / (1j * 2.0 * np.pi * float(f) * float(L))
    return _shunt_y(Y_L, float(Z0))


# =====================================================================
# 4. TRỞ KHÁNG NỐI ĐẤT - MẠNG 1 CỔNG (dùng khi netlist có nút '0'/'GND')
# =====================================================================

def load_1port(Z, Z0=50.0):
    """
    Trở kháng Z nối từ một nút xuống đất, nhìn như mạng 1 cổng:
        Gamma = (Z - Z0) / (Z + Z0)
    Z = np.inf -> Gamma = +1 (hở mạch), Z = 0 -> Gamma = -1 (ngắn mạch).
    """
    if np.isinf(Z):
        g = 1.0 + 0j
    else:
        Z = complex(Z)
        g = (Z - Z0) / (Z + Z0)
    return np.array([[g]], dtype=complex)


def z_of(kind, value, f):
    """Trở kháng của R, L, C tại tần số f (dùng cho load_1port)."""
    w = 2.0 * np.pi * float(f)
    if kind == 'R':
        return complex(float(value))
    if kind == 'L':
        return 1j * w * float(value)
    if kind == 'C':
        return 1.0 / (1j * w * float(value))
    raise ValueError(f"Loại phần tử không hợp lệ: {kind}")


# =====================================================================
# 5. ĐƯỜNG TRUYỀN SÓNG - MẠNG 2 CỔNG
# =====================================================================

def tline_ideal(theta):
    """
    Đường truyền lý tưởng có Zc = Z0 tham chiếu: không phản xạ, chỉ quay pha.
        S11 = S22 = 0,  S21 = S12 = exp(-j*theta)   (theta = beta*l, radian)
    """
    s21 = np.exp(-1j * float(theta))
    return np.array([[0.0, s21],
                     [s21, 0.0]], dtype=complex)


def tline_general(theta, Zc, Z0=50.0):
    """
    Đường truyền không tổn hao có trở kháng đặc tính Zc trong hệ chuẩn Z0.
    Tính qua ABCD: A = D = cos(theta), B = j*Zc*sin(theta), C = j*sin(theta)/Zc
    rồi chuyển sang S (Pozar, Bảng 4.2).
    """
    theta = float(theta)
    Zc = float(Zc)
    Z0 = float(Z0)

    A = np.cos(theta)
    D = A
    B = 1j * Zc * np.sin(theta)
    C = 1j * np.sin(theta) / Zc

    den = A + (B / Z0) + (C * Z0) + D
    s11 = (A + (B / Z0) - (C * Z0) - D) / den
    s12 = 2.0 * (A * D - B * C) / den
    s21 = 2.0 / den
    s22 = (-A + (B / Z0) - (C * Z0) + D) / den

    return np.array([[s11, s12],
                     [s21, s22]], dtype=complex)


def beta_l_from_freq(f, f0, electrical_length_deg=90.0):
    """
    Góc dài điện beta*l (radian) tại tần số f, biết độ dài điện tại f0.
    Đường TEM không tán sắc: theta(f) = theta0 * f / f0.
    """
    theta0 = np.deg2rad(float(electrical_length_deg))
    return theta0 * (float(f) / float(f0))


# =====================================================================
# 6. ĐƯỜNG VI DẢI (MICROSTRIP) - Hammerstad & Jensen (1980), t = 0
# =====================================================================

def microstrip_z0_eeff(W, h, er):
    """
    Trở kháng đặc tính Zc và hằng số điện môi hiệu dụng eps_eff của vi dải
    (mô hình chuẩn tĩnh, độ dày dải t = 0, không tán sắc).
        W  : độ rộng dải dẫn (m)
        h  : độ dày đế (m)
        er : hằng số điện môi tương đối của đế
    """
    u = float(W) / float(h)
    er = float(er)
    a = (1.0 + np.log((u**4 + (u / 52.0)**2) / (u**4 + 0.432)) / 49.0
         + np.log(1.0 + (u / 18.1)**3) / 18.7)
    b = 0.564 * ((er - 0.9) / (er + 3.0))**0.053
    eps_eff = (er + 1.0) / 2.0 + (er - 1.0) / 2.0 * (1.0 + 10.0 / u)**(-a * b)
    fu = 6.0 + (2.0 * np.pi - 6.0) * np.exp(-(30.666 / u)**0.7528)
    z_air = ETA0 / (2.0 * np.pi) * np.log(fu / u + np.sqrt(1.0 + (2.0 / u)**2))
    Zc = z_air / np.sqrt(eps_eff)
    return Zc, eps_eff


def microstrip(W, h, er, length, f, Z0=50.0):
    """
    Ma trận S của đoạn vi dải dài 'length' (m) tại tần số f.
    theta = beta*l với beta = 2*pi*f*sqrt(eps_eff)/c, rồi dùng tline_general.
    """
    Zc, eps_eff = microstrip_z0_eeff(W, h, er)
    beta = 2.0 * np.pi * float(f) * np.sqrt(eps_eff) / C0
    return tline_general(beta * float(length), Zc, Z0)


# =====================================================================
# 7. ĐƯỜNG GHÉP ĐỐI XỨNG (COUPLED LINE) - MẠNG 4 CỔNG
# =====================================================================

def coupled_line(Z0e, Z0o, theta, Z0=50.0, theta_o=None):
    """
    Hai đường truyền ghép đối xứng, phân tích chế độ chẵn / lẻ.
    Đánh số cổng: 1-2 là hai đầu dải a, 3-4 là hai đầu dải b,
                  cổng 1 và cổng 3 cùng một phía.
        S = [[ (Se+So)/2 , (Se-So)/2 ],
             [ (Se-So)/2 , (Se+So)/2 ]]
    theta   : độ dài điện chế độ chẵn (rad)
    theta_o : độ dài điện chế độ lẻ (rad), mặc định = theta (môi trường TEM)
    Bộ ghép lambda/4: hệ số ghép tại f0 là C = (Z0e - Z0o) / (Z0e + Z0o).
    """
    if theta_o is None:
        theta_o = theta
    Se = tline_general(theta, Z0e, Z0)
    So = tline_general(theta_o, Z0o, Z0)
    A = (Se + So) / 2.0
    B = (Se - So) / 2.0
    S = np.zeros((4, 4), dtype=complex)
    S[0:2, 0:2] = A
    S[2:4, 2:4] = A
    S[0:2, 2:4] = B
    S[2:4, 0:2] = B
    return S


# =====================================================================
# 8. ĐIỂM KẾT THÚC, STUB VÀ MỐI NỐI
# =====================================================================

def open_circuit():
    """Hở mạch (mạng 1 cổng): Gamma = +1."""
    return np.array([[1.0 + 0j]], dtype=complex)


def short_circuit():
    """Ngắn mạch (mạng 1 cổng): Gamma = -1."""
    return np.array([[-1.0 + 0j]], dtype=complex)


def stub(theta, Zc, Z0=50.0, termination='open'):
    """
    Stub song song (mạng 1 cổng): đoạn đường truyền Zc dài theta kết thúc hở/ngắn mạch.
        Hở mạch : Zin = -j*Zc*cot(theta)     Ngắn mạch: Zin = j*Zc*tan(theta)
    """
    theta = float(theta)
    if termination == 'open':
        Zin = -1j * Zc / np.tan(theta)
    elif termination == 'short':
        Zin = 1j * Zc * np.tan(theta)
    else:
        raise ValueError("termination phải là 'open' hoặc 'short'")
    return load_1port(Zin, Z0)


def junction_n(n):
    """
    Mối nối lý tưởng N cổng (cùng điện áp, tổng dòng = 0, cùng Z0):
        S = (2/N) * J - I     (J: ma trận toàn số 1)
    N = 3 -> S_ii = -1/3, S_ij = 2/3.
    """
    n = int(n)
    if n < 2:
        raise ValueError("Mối nối cần ít nhất 2 cổng")
    return (2.0 / n) * np.ones((n, n), dtype=complex) - np.eye(n, dtype=complex)


def junction_t():
    """Mối nối ngã ba chữ T lý tưởng 3 cổng: S_ii = -1/3, S_ij = 2/3."""
    return junction_n(3)



# =====================================================================
# 9. BIẾN ĐỔI VÀ KIỂM TRA MA TRẬN (phần chung của nhóm - main.py, solver.py dùng)
# =====================================================================

def z_to_s_Nport(Z, Z0=50.0):
    """Ma trận trở kháng Z (NxN) -> S, tham chiếu thực Z0 chung: S = (Z - Z0 I)(Z + Z0 I)^-1."""
    Z = np.asarray(Z, dtype=complex)
    I = np.eye(Z.shape[0])
    return (Z - Z0 * I) @ np.linalg.inv(Z + Z0 * I)


def s_to_z_Nport(S, Z0=50.0):
    """S -> Z, tham chiếu thực Z0 chung: Z = Z0 (I + S)(I - S)^-1 (không tồn tại nếu I - S suy biến)."""
    S = np.asarray(S, dtype=complex)
    I = np.eye(S.shape[0])
    return Z0 * (I + S) @ np.linalg.inv(I - S)


def renormalize_general(S, z_old, z_new):
    """
    Chuẩn hóa lại S theo sóng công suất (Kurokawa), Z0 phức và khác nhau từng cổng.
        a = (V + Z I) / (2 sqrt(Re Z)),  b = (V - Z* I) / (2 sqrt(Re Z))
    S' = Q P^-1,  P = F'[F^-1 + (G'-G) K (I-S)],  Q = F'[F^-1 - (G+G'*) K (I-S)],
    K = (G+G*)^-1 F^-1. Không đi qua ma trận Z nên dùng được cả cho mối nối T, hở mạch.
    z_old, z_new: số (dùng chung cho mọi cổng) hoặc danh sách N giá trị.
    """
    S = np.asarray(S, dtype=complex)
    n = S.shape[0]
    g = np.broadcast_to(np.asarray(z_old, dtype=complex), (n,))
    gp = np.broadcast_to(np.asarray(z_new, dtype=complex), (n,))
    Finv = np.diag(2.0 * np.sqrt(g.real))
    Fp = np.diag(1.0 / (2.0 * np.sqrt(gp.real)))
    K = np.diag(2.0 * np.sqrt(g.real) / (g + g.conj()))
    I = np.eye(n)
    P = Fp @ (Finv + np.diag(gp - g) @ K @ (I - S))
    Q = Fp @ (Finv - np.diag(g + gp.conj()) @ K @ (I - S))
    return Q @ np.linalg.inv(P)


def renormalize(S, z_old, z_new):
    """Chuẩn hóa lại với Z0 THỰC (trường hợp riêng của renormalize_general)."""
    if np.iscomplexobj(z_old) and np.any(np.imag(z_old)) or np.iscomplexobj(z_new) and np.any(np.imag(z_new)):
        raise ValueError("renormalize chỉ dùng cho Z0 thực; dùng renormalize_general cho Z0 phức")
    return renormalize_general(S, z_old, z_new)


def is_reciprocal(S, tol=1e-9):
    """Thuận nghịch: S = S^T."""
    S = np.asarray(S)
    return bool(np.allclose(S, S.T, atol=tol))


def is_lossless(S, tol=1e-9):
    """Không tổn hao: S^H S = I."""
    S = np.asarray(S)
    return bool(np.allclose(S.conj().T @ S, np.eye(S.shape[0]), atol=tol))


def is_passive(S, tol=1e-9):
    """Thụ động: I - S^H S bán xác định dương (mọi trị riêng >= 0)."""
    S = np.asarray(S)
    M = np.eye(S.shape[0]) - S.conj().T @ S
    return bool(np.linalg.eigvalsh((M + M.conj().T) / 2).min() >= -tol)


# =====================================================================
# BỘ TEST TỰ ĐỘNG (CHẠY TRỰC TIẾP FILE ĐỂ KIỂM TRA)
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("KIỂM THỬ THƯ VIỆN LINH KIỆN COMPONENTS.PY (NGƯỜI B)")
    print("=" * 60)

    f_test = 2.4e9
    Z0 = 50.0

    # 0. to_si
    cases = {'10n': 1e-8, '2.4GHz': 2.4e9, '1Meg': 1e6, '100MHz': 1e8, '70.71': 70.71,
             '1pF': 1e-12, '4.7uH': 4.7e-6, '5mm': 5e-3, '3.3k': 3.3e3, '1f': 1e-15}
    for s, v in cases.items():
        assert np.isclose(to_si(s), v, rtol=1e-12), f"Lỗi to_si({s})"
    print("[PASS] 0. to_si:", len(cases), "trường hợp")

    # 1. Điện trở
    sr = resistor_series(100.0, Z0)
    assert np.isclose(sr[0, 0], 0.5) and np.isclose(sr[0, 1], 0.5), "Lỗi resistor_series"
    sh = resistor_shunt(50.0, Z0)
    assert np.isclose(sh[0, 0], -1/3) and np.isclose(sh[0, 1], 2/3), "Lỗi resistor_shunt"
    print("[PASS] 1. Điện trở nối tiếp và song song")

    # 2. L, C không tổn hao, thuận nghịch
    for S in (capacitor_series(1e-12, f_test), capacitor_shunt(1e-12, f_test),
              inductor_series(2e-9, f_test), inductor_shunt(2e-9, f_test)):
        assert np.allclose(S.conj().T @ S, np.eye(2)), "Lỗi không tổn hao L/C"
        assert np.allclose(S, S.T), "Lỗi thuận nghịch L/C"
    print("[PASS] 2. Tụ điện và cuộn cảm (không tổn hao, thuận nghịch)")

    # 3. Đường truyền
    th = beta_l_from_freq(f_test, f_test, 90)
    assert np.isclose(tline_ideal(th)[0, 1], -1j), "Lỗi tline_ideal"
    tg = tline_general(th, 50 * np.sqrt(2), Z0)
    assert np.isclose(abs(tg[0, 0]), 1/3) and np.isclose(tg[1, 0], -1j * 2 * np.sqrt(2) / 3), "Lỗi tline_general"
    print("[PASS] 3. Đường truyền lý tưởng và tổng quát (70.71 Ohm, lambda/4)")

    # 4. Vi dải: W ~ 3.05 mm trên FR4 h=1.6 mm cho ~50 Ohm
    Zc, ee = microstrip_z0_eeff(3.05e-3, 1.6e-3, 4.4)
    assert 48 < Zc < 52 and 3.2 < ee < 3.5, f"Lỗi microstrip ({Zc:.2f} Ohm, eeff={ee:.3f})"
    S = microstrip(3.05e-3, 1.6e-3, 4.4, 10e-3, f_test)
    assert np.allclose(S.conj().T @ S, np.eye(2)), "Lỗi vi dải không tổn hao"
    print(f"[PASS] 4. Vi dải W=3.05mm, h=1.6mm, er=4.4: Zc={Zc:.2f} Ohm, eeff={ee:.3f}")

    # 5. Đường ghép lambda/4, C = 10 dB
    Cc = 10 ** (-10 / 20)
    Z0e, Z0o = Z0 * np.sqrt((1 + Cc) / (1 - Cc)), Z0 * np.sqrt((1 - Cc) / (1 + Cc))
    Scl = coupled_line(Z0e, Z0o, np.pi / 2, Z0)
    assert np.isclose(abs(Scl[2, 0]), Cc), "Lỗi hệ số ghép"
    assert abs(Scl[0, 0]) < 1e-12 and abs(Scl[3, 0]) < 1e-12, "Lỗi phối hợp/cách ly"
    assert np.allclose(Scl.conj().T @ Scl, np.eye(4)), "Lỗi đường ghép không tổn hao"
    print(f"[PASS] 5. Đường ghép lambda/4: |S31| = {abs(Scl[2,0]):.4f} (= C 10 dB), S11 = S41 = 0")

    # 6. Hở, ngắn mạch, stub, mối nối
    assert np.isclose(open_circuit()[0, 0], 1) and np.isclose(short_circuit()[0, 0], -1)
    assert np.isclose(stub(np.pi / 2, 50, Z0, 'open')[0, 0], -1), "Lỗi stub hở lambda/4"
    j3 = junction_t()
    assert np.isclose(j3[0, 0], -1/3) and np.isclose(j3[0, 1], 2/3), "Lỗi junction_t"
    for n in (3, 4, 5):
        J = junction_n(n)
        assert np.allclose(J.conj().T @ J, np.eye(n)), f"Lỗi junction_n({n})"
    print("[PASS] 6. Hở mạch, ngắn mạch, stub, mối nối T và N cổng")

    print("\n=> TẤT CẢ KIỂM THỬ ĐẠT.")
