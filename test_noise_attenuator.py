"""Analytical 3 dB matched T attenuator test for noise_analysis.py.
Copy this and attenuator_3db.net next to noise_analysis.py and run:
    python3 test_noise_attenuator.py
"""
from pathlib import Path
import numpy as np
from scipy.constants import Boltzmann as KB
from netlist import read_netlist
from noise_analysis import analyze_passive_noise, T0

root = Path(__file__).resolve().parent
net = read_netlist(str(root / 'attenuator_3db.net'))
f = 1e9
result = analyze_passive_noise(net, f, temperature=T0)
S, C = result['S'], result['C']
gain = abs(S[1,0])**2
expected_gain = 10**(-3/10)
expected_c22 = KB*T0*(1-expected_gain)
expected_F = 1/expected_gain
error = np.linalg.norm((C-KB*T0*(np.eye(2)-S@S.conj().T))/(KB*T0))
print('3 dB T attenuator at f = 1 GHz, T=290 K, Z0=50 ohm')
print('S =\n', S)
print(f'S11 = {abs(S[0,0]):.6e}, S22 = {abs(S[1,1]):.6e}')
print(f'|S21|^2 = {gain:.12f}; expected = {expected_gain:.12f}')
print(f'Insertion loss = {-10*np.log10(gain):.9f} dB')
print(f'C22 = {C[1,1].real:.12e} W/Hz; expected = {expected_c22:.12e} W/Hz')
print(f'NF = {result["NF_dB"]:.9f} dB; expected = 3.000000000 dB')
print(f'Covariance residual normalized to kBT0: {error:.3e}')
assert abs(S[0,0])<1e-10 and abs(S[1,1])<1e-10
assert abs(gain-expected_gain)<1e-10
assert abs(result['NF_dB']-3)<1e-9
assert abs(C[1,1].real-expected_c22)<1e-12*KB*T0
assert error<1e-10
print('PASS: analytical attenuation + noise + covariance tests')
