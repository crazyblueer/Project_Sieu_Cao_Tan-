"""Validated noise-analysis adapter for GUI v5 (S-matrix / noise-wave units).

Passive circuits: correlated thermal noise C=k_B*T*(I-S*S^H), and unused
external ports are explicitly matched with thermal terminations at T.
Active two-ports: requires USER-ENABLED assumed/measured S2P noise parameters.
No claim that Touchstone S2P data alone contains noise parameters.
Source is matched at T0=290 K; destination is matched/noiseless.
All ports use a common real 50-ohm reference impedance.
"""
from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann as KB
from solver import solve_netlist_at_freq
from noise_analysis import analyze_passive_noise, T0
from noise_active import analyze_amplifier


def _positive(value, label):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{label} must be a finite positive number')
    return value


def _db(z):
    return float(20*np.log10(max(abs(z), 1e-15)))


def evaluate_noise_at(net, frequency_hz, source_port, output_port,
                      temperature_k=T0, active_parameters=None):
    """Compute matched-source noise figure for selected external port pair.

    Other ports of a passive multiport are matched at temperature_k;
    their thermal noise is INCLUDED in equivalent output C.
    active_parameters: explicit mapping {S2P element name: noise parameters}.
    Active case supports exactly 2 ports and source P1 -> destination P2.
    """
    frequency_hz = _positive(frequency_hz, 'Frequency')
    temperature_k = _positive(temperature_k, 'Temperature (K)')
    ports = sorted(net.external_ports)
    if source_port not in ports or output_port not in ports or source_port == output_port:
        raise ValueError('Source and output must be two distinct external ports')
    if len(ports) < 2:
        raise ValueError('A noise figure requires at least two external ports')
    names = [e['name'] for e in net.elements if e['type'] == 'S2P']
    i, o = ports.index(source_port), ports.index(output_port)
    if names:
        if len(ports) != 2 or (source_port, output_port) != tuple(ports):
            raise ValueError('Active S2P noise currently supports only 2-port forward P1→P2')
        if len(names) != 1:
            raise ValueError('GUI v5 supports one active S2P device per network; separate noise models are needed for multiple devices')
        if not active_parameters or any(n not in active_parameters for n in names):
            raise ValueError('Missing explicit S2P noise parameters. Enable the assumed model or supply measured data.')
        result = analyze_amplifier(net, frequency_hz, active_parameters,
                                   temperature=temperature_k)
        s, c = result['S'], result['C']
        intrinsic = float(np.real(c[o, o]))
        terminated = 0.0
        f = float(result['F'])
        model = 'ACTIVE — S2P noise parameters supplied separately (illustrative if assumed)'
    else:
        result = analyze_passive_noise(net, frequency_hz, temperature=temperature_k)
        s, c = result['S'], result['C']
        if s.shape != (len(ports),len(ports)):
            raise ValueError('Unexpected S-matrix order')
        intrinsic = float(np.real(c[o, o]))
        unused = [k for k in range(len(ports)) if k not in (i,o)]
        terminated = float(KB*temperature_k*sum(abs(s[o,k])**2 for k in unused))
        model = ('PASSIVE — isothermal thermal noise; '
                 f'{len(unused)} unused port(s) matched at {temperature_k:g} K')
        gain = abs(s[o,i])**2
        if gain <= 1e-25:
            raise ValueError('Selected transfer has ~zero gain; noise figure is undefined/infinite')
        f = 1 + (intrinsic + terminated)/(KB*T0*gain)
    gain = abs(s[o,i])**2
    if not np.isfinite(f) or f <= 0:
        raise ValueError('Noise figure result is invalid or nonfinite')
    return {
        'frequency_hz': float(frequency_hz),
        'source_port': int(source_port), 'output_port': int(output_port),
        'temperature_k': float(temperature_k), 'reference_temperature_k': float(T0),
        'nf_db': float(10*np.log10(f)), 'f_linear': float(f),
        'gain_db': _db(s[o,i]), 'gain_linear': float(gain),
        'intrinsic_cout_w_per_hz': intrinsic,
        'termination_cout_w_per_hz': terminated,
        'total_cout_w_per_hz': intrinsic + terminated,
        'model': model, 'active': bool(names),
    }


def evaluate_noise_sweep(net, frequencies_hz, source_port, output_port,
                         temperature_k=T0, active_parameters=None):
    freqs = np.asarray(frequencies_hz, dtype=float)
    if freqs.ndim != 1 or not (1 <= len(freqs) <= 2001):
        raise ValueError('Noise sweep must contain 1–2001 frequencies')
    if not np.all(np.isfinite(freqs)) or np.any(freqs <= 0):
        raise ValueError('All noise frequencies must be positive and finite')
    return [evaluate_noise_at(net, float(f), source_port, output_port,
                              temperature_k, active_parameters) for f in freqs]


def branchline_hybrid_check(net, frequency_hz=2.4e9):
    """Assess *4-port branch-line* targets; not a generic 4-port performance spec.

    Port definitions: P1 input, P2 through, P3 coupled, P4 isolated.
    Acceptance gates are for a narrow-band, ideal 3 dB coupler at its f0.
    """
    if sorted(net.external_ports) != [1, 2, 3, 4]:
        raise ValueError('Branch-line check requires PORT 1, PORT 2, PORT 3, PORT 4')
    s = np.asarray(solve_netlist_at_freq(net, _positive(frequency_hz,'Frequency'),
                                         method='matrix'), complex)
    if s.shape != (4,4):
        raise ValueError('Expected a 4×4 S-matrix')
    coupling = -10*np.log10(2)
    s11, s21, s31, s41 = [_db(s[k,0]) for k in (0,1,2,3)]
    phase21 = float(np.angle(s[1,0],deg=True))
    phase31 = float(np.angle(s[2,0],deg=True))
    phase_diff = float(np.angle(s[1,0]/s[2,0],deg=True))
    quadrature_error = float(abs(abs(phase_diff)-90))
    unitarity = float(np.linalg.norm(s.conj().T@s-np.eye(4)))
    passing = (s11 < -20 and s41 < -20 and abs(s21-coupling)<0.15
               and abs(s31-coupling)<0.15 and quadrature_error<5
               and unitarity<1e-4)
    return {'frequency_hz':float(frequency_hz), 's11_db':s11, 's21_db':s21,
            's31_db':s31,'s41_db':s41,'phase21_deg':phase21,
            'phase31_deg':phase31,'phase_difference_deg':phase_diff,
            'quadrature_error_deg':quadrature_error,
            'unitarity_residual':unitarity,
            'pass':bool(passing)}
