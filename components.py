import numpy as np
import re

def to_si(val):

    if isinstance(val, (int, float)):
        return float(val)
        
    val_str = str(val).strip().lower()
    
    multipliers = {
        'f': 1e-15,  # femto
        'p': 1e-12,  # pico (pF)
        'n': 1e-9,   # nano (nH)
        'u': 1e-6,   # micro (uF)
        'm': 1e-3,   # milli
        'k': 1e3,    # kilo
        'meg': 1e6,  # Mega
        'mhz': 1e6,  # Mega (Hz)
        'g': 1e9,    # Giga (GHz)
        'ghz': 1e9
    }
    
    match = re.match(r"^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([a-zA-Z]*)$", val_str)
    if not match:
        raise ValueError(f"Không thể nhận diện định dạng giá trị: {val}")
        
    number, unit = match.groups()
    number = float(number)
    
    unit = unit.replace('hz', '').replace('ohm', '').replace('f', '').replace('h', '')
    
    for prefix, factor in multipliers.items():
        if unit == prefix or unit.startswith(prefix):
            return number * factor
            
    return number

def resistor_series(R, Z0=50.0):
    """
        S11 = S22 = R / (R + 2*Z0)
        S21 = S12 = 2*Z0 / (R + 2*Z0)
    """
    R = float(R)
    Z0 = float(Z0)
    den = R + 2.0 * Z0
    s11 = R / den
    s21 = 2.0 * Z0 / den
    return np.array([
        [s11, s21],
        [s21, s11]
    ], dtype=complex)


def resistor_shunt(R, Z0=50.0):
    """
        Y = 1 / R:
        S11 = S22 = -Z0 / (Z0 + 2*R)
        S21 = S12 = 2*R / (Z0 + 2*R)
    """
    R = float(R)
    Z0 = float(Z0)
    den = 2.0 * R + Z0
    s11 = -Z0 / den
    s21 = (2.0 * R) / den
    return np.array([
        [s11, s21],
        [s21, s11]
    ], dtype=complex)


def capacitor_series(C, f, Z0=50.0):
    """
        Z_C = 1 / (j * 2*pi*f * C)
    """
    omega = 2.0 * np.pi * float(f)
    Z_C = 1.0 / (1j * omega * float(C))
    den = Z_C + 2.0 * Z0
    s11 = Z_C / den
    s21 = (2.0 * Z0) / den
    return np.array([
        [s11, s21],
        [s21, s11]
    ], dtype=complex)


def capacitor_shunt(C, f, Z0=50.0):
    """
        Y_C = j * 2*pi*f * C
    """
    omega = 2.0 * np.pi * float(f)
    Y_C = 1j * omega * float(C)
    den = Y_C * Z0 + 2.0
    s11 = -(Y_C * Z0) / den
    s21 = 2.0 / den
    return np.array([
        [s11, s21],
        [s21, s11]
    ], dtype=complex)



def inductor_series(L, f, Z0=50.0):
    """
        Z_L = j * 2*pi*f * L
    """
    omega = 2.0 * np.pi * float(f)
    Z_L = 1j * omega * float(L)
    den = Z_L + 2.0 * Z0
    s11 = Z_L / den
    s21 = (2.0 * Z0) / den
    return np.array([
        [s11, s21],
        [s21, s11]
    ], dtype=complex)


def inductor_shunt(L, f, Z0=50.0):
    """
        Y_L = 1 / (j * 2*pi*f * L)
    """
    omega = 2.0 * np.pi * float(f)
    Z_L = 1j * omega * float(L)
    Y_L = 1.0 / Z_L
    den = Y_L * Z0 + 2.0
    s11 = -(Y_L * Z0) / den
    s21 = 2.0 / den
    return np.array([
        [s11, s21],
        [s21, s11]
    ], dtype=complex)



def tline_ideal(theta):
    """
        S11 = S22 = 0
        S21 = S12 = exp(-j * theta)
    """
    s21 = np.exp(-1j * float(theta))
    return np.array([
        [0.0,  s21],
        [s21,  0.0]
    ], dtype=complex)


def tline_general(theta, Zc, Z0=50.0):
    """
        A = D = cos(theta)
        B = j*Zc*sin(theta)
        C = j*sin(theta)/Zc
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
    
    return np.array([
        [s11, s12],
        [s21, s22]
    ], dtype=complex)


def beta_l_from_freq(f, f0, electrical_length_deg=90.0):

    theta0 = np.deg2rad(electrical_length_deg)
    return theta0 * (f / f0)


def open_circuit():
    """
       Gamma = +1
    """
    return np.array([[1.0 + 0j]], dtype=complex)


def short_circuit():
    """
        Gamma = -1
    """
    return np.array([[-1.0 + 0j]], dtype=complex)


def junction_t():
    """
        S_ii = -1/3
        S_ij = 2/3
    """
    return np.array([
        [-1.0/3.0,  2.0/3.0,  2.0/3.0],
        [ 2.0/3.0, -1.0/3.0,  2.0/3.0],
        [ 2.0/3.0,  2.0/3.0, -1.0/3.0]
    ], dtype=complex)

#ADD QA
def z_to_s_Nport(Z, Z0=50.0):
    """Chuyen tro khang tai Z sang tham so S (he so phan xa) cho mang N cong.
    Z, Z0 co the la so phuc. Ho Tro Z la ma tran (quet theo tan so)"""
    
    Z = np.asarray(Z, dtype = complex)
    N = Z.shape[0] #so cong 
    I = np.eye(N) 
    S = (Z -Z0*I)  @ np.linalg.inv(Z + Z0*I)
    return S

def s_to_z_Nport(S, Z0=50.0):
    """Nguoc cua z_to_s_Nport: tu S ve Z. Z khong phu thuoc he tham chieu."""
    S = np.asarray(S, dtype=complex)
    N = S.shape[0]
    I = np.eye(N)
    Z = Z0 * np.linalg.inv(I - S) @ (I + S)
    return Z

def renormalize(S, Z0_old, Z0_new):
    """Doi ma tran S tu he tham chieu Z0_old sang Z0_new (di vong qua Z bat bien)."""
    Z = s_to_z_Nport(S, Z0_old)      # buoc 1: ve su that vat ly Z
    S_new = z_to_s_Nport(Z, Z0_new)  # buoc 2: nhin tu he moi
    return S_new

#TEST RENORMALIZATION HAM PHUC 
def _FG(z0_vec):
    z0 = np.asarray(z0_vec, dtype=complex)
    F  = np.diag(1.0/np.sqrt(np.real(z0)))   # Re(z0) phai > 0
    G  = np.diag(z0)
    Gc = np.diag(np.conj(z0))                 # Z0* (lien hop) - mau chot cua Kurokawa
    return F, G, Gc

def z_to_s_general(Z, z0_vec):
    """Z -> S, power wave, Z0 phuc / khac nhau tung cong."""
    Z = np.asarray(Z, dtype=complex)
    F, G, Gc = _FG(z0_vec)
    return F @ (Z - Gc) @ np.linalg.inv(Z + G) @ np.linalg.inv(F)

def s_to_z_general(S, z0_vec):
    """S -> Z (nguoc lai)."""
    S = np.asarray(S, dtype=complex); N = S.shape[0]; I = np.eye(N)
    F, G, Gc = _FG(z0_vec)
    Sp = np.linalg.inv(F) @ S @ F
    return np.linalg.inv(I - Sp) @ (Gc + Sp @ G)

def renormalize_general(S, z0_old, z0_new):
    """Doi he tham chieu, cho phep Z0 phuc va khac nhau tung cong."""
    N = S.shape[0]
    z0_old = np.broadcast_to(np.asarray(z0_old, dtype=complex), (N,))
    z0_new = np.broadcast_to(np.asarray(z0_new, dtype=complex), (N,))
    Z = s_to_z_general(S, z0_old)
    return z_to_s_general(Z, z0_new)


 #ham kiem tra tinh chat mang 

def is_reciprocal(S, tol=1e-9):
    """Thuan nghich: S = S^T (doi xung). Dung cho mach thu dong thong thuong."""
    S = np.asarray(S, dtype=complex)
    return np.allclose(S, S.T, atol=tol)

def is_lossless(S, tol=1e-9):
    """Khong ton hao: S unita, tuc S^H @ S = I (S^H = chuyen vi lien hop)."""
    S = np.asarray(S, dtype=complex)
    N = S.shape[0]
    return np.allclose(S.conj().T @ S, np.eye(N), atol=tol)
def is_passive(S, tol=1e-9):
    """Thu dong: ko tu sinh nang luong. I - S^H @ S co moi gia tri rieng >= 0."""
    S = np.asarray(S, dtype=complex)
    N = S.shape[0]
    M = np.eye(N) - S.conj().T @ S
    eigenvalues = np.linalg.eigvalsh(M)   # eigvalsh cho ma tran Hermit, tra ve gia tri thuc
    return np.all(eigenvalues >= -tol)



if __name__ == "__main__":
    # Test renormalize: doan day 70.7 ohm he cu -> doi ve he 50 ohm
    S_old = tline_ideal(np.pi/2)                      # [[0,-j],[-j,0]] trong he 70.7
    S_new = renormalize(S_old, 50*np.sqrt(2), 50)     # doi ve 50 ohm
    expected = tline_general(np.pi/2, 50*np.sqrt(2), 50)
    assert np.allclose(S_new, expected), "renormalize SAI!"
    print("renormalize OK")
    
    # Doan day ly tuong: thuan nghich, KHONG ton hao (day ly tuong ko mat nang luong)
    line = tline_ideal(np.pi/2)
    print("Day ly tuong:")
    print("  reciprocal:", is_reciprocal(line))   # True
    print("  lossless  :", is_lossless(line))      # True
    print("  passive   :", is_passive(line))       # True

# Dien tro noi tiep: thuan nghich, CO ton hao (dien tro an nang luong) -> ko lossless, nhung passive
    R = resistor_series(50, 50)
    print("Dien tro noi tiep:")
    print("  reciprocal:", is_reciprocal(R))       # True
    print("  lossless  :", is_lossless(R))          # False (co ton hao!)
    print("  passive   :", is_passive(R))           # True (ko tu sinh nang luong)

# Wilkinson (lay tu ket qua S cua ban) - nen thuan nghich, co ton hao tren R cach ly
# print(is_reciprocal(S_wilkinson), is_lossless(S_wilkinson), is_passive(S_wilkinson))
