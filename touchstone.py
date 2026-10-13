import numpy as np


class Touchstone2P:
    
    def __init__(self, filename=None):
        self.filename = filename
        self.freqs = np.array([])        
        self.s_matrices = np.array([])   
        self.z0 = 50.0                   
        self.freq_unit = "HZ"
        self.data_format = "MA"          

        if filename is not None:
            self.load(filename)

    def load(self, filename):
        self.filename = filename
        freq_list = []
        s_list = []

        freq_multipliers = {
            'HZ': 1.0,
            'KHZ': 1e3,
            'MHZ': 1e6,
            'GHZ': 1e9
        }

        mult = 1.0
        fmt = "MA"
        z0_val = 50.0

        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith('!'):
                    continue

                if line_str.startswith('#'):
                    tokens = line_str[1:].strip().upper().split()
                    for i, tok in enumerate(tokens):
                        if tok in freq_multipliers:
                            mult = freq_multipliers[tok]
                            self.freq_unit = tok
                        elif tok in ["MA", "DB", "RI"]:
                            fmt = tok
                            self.data_format = tok
                        elif tok == "R" and i + 1 < len(tokens):
                            z0_val = float(tokens[i + 1])
                            self.z0 = z0_val
                    continue

                tokens = line_str.split()

                if len(tokens) >= 9:
                    vals = [float(x) for x in tokens[:9]]
                    f_hz = vals[0] * mult
                    freq_list.append(f_hz)

                    s11 = self._parse_s_pair(vals[1], vals[2], fmt)
                    s21 = self._parse_s_pair(vals[3], vals[4], fmt)
                    s12 = self._parse_s_pair(vals[5], vals[6], fmt)
                    s22 = self._parse_s_pair(vals[7], vals[8], fmt)

                    s_mat = np.array([
                        [s11, s12],
                        [s21, s22]
                    ], dtype=complex)
                    s_list.append(s_mat)

        self.freqs = np.array(freq_list, dtype=float)
        self.s_matrices = np.array(s_list, dtype=complex)
        return self

    @staticmethod
    def _parse_s_pair(val1, val2, fmt):

        if fmt == "MA":
            mag = val1
            ang_rad = np.deg2rad(val2)
            return mag * (np.cos(ang_rad) + 1j * np.sin(ang_rad))
        elif fmt == "DB":
            mag = 10.0 ** (val1 / 20.0)
            ang_rad = np.deg2rad(val2)
            return mag * (np.cos(ang_rad) + 1j * np.sin(ang_rad))
        elif fmt == "RI":
            return complex(val1, val2)
        else:
            raise ValueError(f"Định dạng {fmt} không hợp lệ.")

    def s_interp(self, f_target):
        
        if len(self.freqs) == 0:
            raise ValueError("No data")

        s_target = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                real_part = np.interp(f_target, self.freqs, self.s_matrices[:, i, j].real)
                imag_part = np.interp(f_target, self.freqs, self.s_matrices[:, i, j].imag)
                s_target[i, j] = real_part + 1j * imag_part

        return s_target


def load_s2p(filename):
    
    return Touchstone2P(filename)

