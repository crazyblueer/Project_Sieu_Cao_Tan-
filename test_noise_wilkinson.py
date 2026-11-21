"""Topic 7: noise of a Wilkinson 3-port with matched 50-ohm termination.

Port 1: driven by a matched source at reference temperature 290 K.
Port 2: measured output (matched noiseless measurement receiver).
Port 3: matched 50-ohm termination at 290 K.
All noise PSDs are single-sided power spectral densities (W/Hz).
Requires noise_analysis.py, netlist.py and their original dependencies.
"""
from pathlib import Path
import csv
import numpy as np
from scipy.constants import Boltzmann as KB
from netlist import read_netlist
from noise_analysis import analyze_passive_noise, T0

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results_noise'
RESULTS.mkdir(exist_ok=True)


def terminated_noise_twoport(S, C, keep=(0, 1), terminated=(2,), temperature=T0):
    """Matched termination: incident thermal noise kBT; no reflection.

    Assumes real equal 50 ohm references, uncorrelated matched termination
    noise, and source not included in intrinsic C. Returns the reduced 2-port
    S and intrinsic output noise C including termination noise.
    """
    S = np.asarray(S, dtype=complex)
    C = np.asarray(C, dtype=complex)
    keep, term = list(keep), list(terminated)
    n = S.shape[0]
    assert S.shape == (n, n) and C.shape == (n, n)
    assert len(set(keep + term)) == n and sorted(keep + term) == list(range(n))
    S2 = S[np.ix_(keep, keep)]
    C2 = C[np.ix_(keep, keep)].copy()
    if term:
        coupling = S[np.ix_(keep, term)]
        C2 += KB * temperature * (coupling @ coupling.conj().T)
    C2 = (C2 + C2.conj().T) / 2
    return S2, C2


def analyze_wilkinson(frequency=2.4e9):
    net = read_netlist(str(ROOT / 'wilkinson.net'))
    data = analyze_passive_noise(net, frequency, T0)
    S, C = data['S'], data['C']
    S2, C2 = terminated_noise_twoport(S, C, keep=(0, 1), terminated=(2,))
    g = abs(S2[1, 0])**2
    F = 1 + C2[1, 1].real / (KB*T0*g)
    nf_db = 10*np.log10(F)
    # The source at 290 K provides kBT*g noise at measured output.
    # This is a model identity for isothermal passive networks, NOT independent validation.
    ceq = KB*T0*(np.eye(2) - S2 @ S2.conj().T)
    residual = np.linalg.norm((C2-ceq)/(KB*T0))
    return dict(S=S, C=C, S2=S2, C2=C2, gain=g, F=F,
                nf_db=nf_db, residual=residual,
                device_noise=C[1,1].real,
                term_noise=KB*T0*abs(S[1,2])**2)


def main():
    f0 = 2.4e9
    x = analyze_wilkinson(f0)
    expected = 10*np.log10(2.0)  # ideal equal splitter at matched ports
    print('Wilkinson 3-port | Port 1 -> Port 2 | Port 3 terminated in 50 ohm at 290 K')
    print(f'frequency = {f0/1e9:.3f} GHz')
    print(f'|S21|^2 = {x["gain"]:.9f} (ideal 0.5)')
    print(f'intrinsic C22 = {x["device_noise"]:.9e} W/Hz')
    print(f'port-3 termination contribution to output = {x["term_noise"]:.9e} W/Hz')
    print(f'effective C22 = {x["C2"][1,1].real:.9e} W/Hz')
    print(f'NF = {x["nf_db"]:.9f} dB (ideal {expected:.9f} dB)')
    print(f'covariance identity residual / (kBT0) = {x["residual"]:.3e}')
    assert x['residual'] < 1e-8
    assert abs(x['nf_db']-expected) < 0.005, 'Unexpected deviation at center frequency'
    print('PASS: matched-terminated Wilkinson center-frequency test')

    freqs = np.linspace(1.8e9, 3.0e9, 241)
    records = []
    for f in freqs:
        v = analyze_wilkinson(f)
        records.append((f/1e9, v['gain'], v['nf_db'],
                        v['device_noise'], v['term_noise'], v['C2'][1,1].real,
                        v['residual']))
    path = RESULTS / 'wilkinson_noise_terminated.csv'
    with path.open('w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['frequency_GHz','available_gain_linear','NF_dB',
                         'intrinsic_C22_W_per_Hz','termination_C22_W_per_Hz',
                         'effective_C22_W_per_Hz','covariance_residual_norm'])
        writer.writerows(records)
    print(f'Sweep: {len(records)} points | NF range: {min(r[2] for r in records):.5f} - {max(r[2] for r in records):.5f} dB')
    print(f'CSV: {path}')
    try:
        import matplotlib.pyplot as plt
        f = [r[0] for r in records]
        plt.figure(figsize=(8,4.6))
        plt.plot(f, [r[2] for r in records], label='Noise Figure')
        plt.axvline(2.4, linestyle='--', alpha=0.5, label='2.4 GHz')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('NF (dB)')
        plt.title('Wilkinson 1 -> 2 | Port 3 matched, 290 K')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plot_path = RESULTS / 'wilkinson_noise_terminated.png'
        plt.savefig(plot_path, dpi=180)
        plt.close()
        print(f'Plot: {plot_path}')
    except ImportError:
        print('Matplotlib not installed; CSV was exported without PNG.')

if __name__ == '__main__':
    main()
