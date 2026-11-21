#!/usr/bin/env python3
"""Test component-tolerance robustness of an optimized LC netlist.

Place next to netlist.py, solver.py, components.py in the team's project.

Example:
    python3 test_optimizer_robustness.py --netlist results_optimization_v2/filter_lc_optimized_v2.net --tolerance 5
    python3 test_optimizer_robustness.py --netlist results_optimization_robust/filter_lc_optimized_v2.net --tolerance 5

This is an independent random-tolerance *scenario test* using the team's
S-matrix solver. It is not an independent reference implementation, nor a
statistical guarantee of production yield. Ideal component models only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from netlist import read_netlist
from solver import solve_netlist_at_freq
from components import to_si

NAMES = ('C1', 'L2', 'C3', 'L4', 'C5')
PB = (0.10e9, 0.75e9)
SB = (1.20e9, 1.60e9)
PASS_MAX_DB = 0.35
STOP_MIN_DB = 22.0


def get_original_components(net):
    found = {}
    for e in net.elements:
        if e['name'] in NAMES:
            if e['type'] not in ('CAP', 'IND'):
                raise ValueError(f"Unexpected type for {e['name']}: {e['type']}")
            key = 'C' if e['type'] == 'CAP' else 'L'
            found[e['name']] = (key, float(to_si(e['params'][key])))
    if set(found) != set(NAMES):
        raise ValueError('Netlist must contain C1, L2, C3, L4, C5')
    return found


def assign_perturbed_values(net, base, multipliers):
    for e in net.elements:
        name = e['name']
        if name in base:
            key, nominal = base[name]
            scale, suffix = (1e-12, 'p') if key == 'C' else (1e-9, 'n')
            value = nominal * multipliers[NAMES.index(name)]
            e['params'][key] = f'{value / scale:.12g}{suffix}'


def band_peak_loss(net, frequencies, mode):
    worst = -np.inf if mode == 'pass' else np.inf
    for f in frequencies:
        s = solve_netlist_at_freq(net, float(f), method='matrix')
        if s.shape != (2, 2) or not np.isfinite(s).all():
            raise ValueError(f'Invalid S-matrix at {f} Hz')
        loss = -20 * np.log10(max(abs(s[1, 0]), 1e-14))
        worst = max(worst, loss) if mode == 'pass' else min(worst, loss)
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--netlist', default='results_optimization_v2/filter_lc_optimized_v2.net')
    ap.add_argument('--tolerance', type=float, default=5, help='Uniform +/- tolerance in percent')
    ap.add_argument('--trials', type=int, default=220)
    ap.add_argument('--points', type=int, default=81, help='Frequency points per band')
    ap.add_argument('--seed', type=int, default=20261121)
    args = ap.parse_args()
    if not (0 < args.tolerance < 100) or args.trials < 1 or args.points < 3:
        ap.error('Set 0 < tolerance < 100, trials >= 1 and points >= 3')
    path = Path(args.netlist).expanduser()
    if not path.is_absolute() and not path.is_file():
        path = Path(__file__).resolve().parent / path
    path = path.resolve()
    if not path.is_file():
        ap.error(f'Netlist not found: {path}')
    net = read_netlist(str(path))
    base = get_original_components(net)
    pass_f = np.linspace(*PB, args.points)
    stop_f = np.linspace(*SB, args.points)
    rng = np.random.default_rng(args.seed)
    pb, sb = np.empty(args.trials), np.empty(args.trials)
    delta = args.tolerance / 100.0
    for i in range(args.trials):
        multipliers = rng.uniform(1 - delta, 1 + delta, size=len(NAMES))
        assign_perturbed_values(net, base, multipliers)
        pb[i] = band_peak_loss(net, pass_f, 'pass')
        sb[i] = band_peak_loss(net, stop_f, 'stop')
    passing = (pb <= PASS_MAX_DB) & (sb >= STOP_MIN_DB)
    print(f'Netlist: {path.name}')
    print(f'Random independent uniform component tolerances: +/-{args.tolerance:g}%')
    print(f'{args.trials} scenarios; {args.points} frequencies per band; seed={args.seed}')
    print(f'Passband condition: loss <= {PASS_MAX_DB:.3f} dB')
    print(f'Stopband condition: attenuation >= {STOP_MIN_DB:.3f} dB')
    print(f'Passband compliance: {(pb <= PASS_MAX_DB).mean() * 100:.2f}%')
    print(f'Stopband compliance: {(sb >= STOP_MIN_DB).mean() * 100:.2f}%')
    print(f'Both specifications: {passing.mean() * 100:.2f}%')
    print(f'95th percentile passband peak loss: {np.percentile(pb, 95):.4f} dB')
    print(f'5th percentile stopband attenuation: {np.percentile(sb, 5):.4f} dB')
    print('NOTE: scenario yield is an estimate; components are ideal except for value variations.')


if __name__ == '__main__':
    main()
