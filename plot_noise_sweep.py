"""Noise-frequency sweep of Topic 7 passive circuits.
Place next to noise_analysis.py, netlist.py, etc. Run: python3 plot_noise_sweep.py
Outputs PNG + CSV files in results_noise/.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann as KB
from netlist import read_netlist
from noise_analysis import analyze_passive_noise, T0

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results_noise'
OUTPUT.mkdir(exist_ok=True)


def sweep(netlist_name, freqs_hz):
    net = read_netlist(str(ROOT / netlist_name))
    rows = []
    for freq in freqs_hz:
        x = analyze_passive_noise(net, float(freq), temperature=T0)
        s, c = x['S'], x['C']
        if s.shape != (2, 2):
            raise ValueError('These plots require a two-port network')
        gain = abs(s[1, 0])**2
        rows.append((freq / 1e9, 20*np.log10(max(abs(s[1, 0]), 1e-300)),
                     20*np.log10(max(abs(s[0, 0]), 1e-300)),
                     x['NF_dB'], float(c[0,0].real), float(c[1,1].real),
                     np.linalg.norm(c-KB*T0*(np.eye(2)-s@s.conj().T))/(KB*T0)))
    a = np.asarray(rows)
    tag = Path(netlist_name).stem
    with (OUTPUT / f'{tag}_noise_sweep.csv').open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['frequency_GHz','S21_dB','S11_dB','NF_dB','C11_W_per_Hz','C22_W_per_Hz','covariance_error_normalized'])
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(a[:,0], a[:,3], label='NF (dB)')
    ax.plot(a[:,0], -a[:,1], '--', label='-S21 (dB): insertion attenuation')
    ax.set(xlabel='Frequency (GHz)', ylabel='dB', title=f'Noise figure: {tag}')
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUT / f'{tag}_NF.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    # Normalize to k_B*T0 for more meaningful plots near floating-point noise floor
    ax.plot(a[:,0], a[:,4]/(KB*T0), label='C11 / (kB*T0)')
    ax.plot(a[:,0], a[:,5]/(KB*T0), label='C22 / (kB*T0)')
    ax.set(xlabel='Frequency (GHz)', ylabel='Normalized noise PSD', title=f'Intrinsic noise covariance: {tag}')
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUT / f'{tag}_C.png', dpi=160); plt.close(fig)

    print(f'{netlist_name}: {len(a)} frequencies; max covariance residual = {np.max(a[:,6]):.3e}')
    print(f'  NF range = [{np.nanmin(a[:,3]):.6f}, {np.nanmax(a[:,3]):.6f}] dB')
    return a

if __name__ == '__main__':
    sweep('attenuator_3db.net', np.linspace(0.1e9, 3e9, 101))
    sweep('filter_lc.net', np.linspace(0.05e9, 3e9, 301))
    print('Saved charts and CSV to:', OUTPUT)
