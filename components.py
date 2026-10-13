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
