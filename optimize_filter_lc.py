"""Optimize symmetric 5th-order LC filter with the team's own S-matrix solver.
Run: python3 optimize_filter_lc.py
Requires numpy, scipy, matplotlib. Netlist must be filter_lc.net beside script.
No edits to the original netlist.
"""
from pathlib import Path
import csv
import numpy as np
from scipy.optimize import differential_evolution
from netlist import read_netlist
from solver import solve_netlist_at_freq

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results_optimization'
NETFILE = ROOT / 'filter_lc.net'
NAMES = {'C1': ('C', 1e-12), 'L2': ('L', 1e-9), 'C3': ('C', 1e-12),
         'L4': ('L', 1e-9), 'C5': ('C', 1e-12)}

# Design targets are a new demonstration specification, not the original professor's requirements.
PASS_FREQ = np.linspace(.1e9, .75e9, 9)
STOP_FREQ = np.linspace(1.2e9, 1.6e9, 9)
PASS_MAX_DB = .35
STOP_MIN_DB = 22.0
F_EVAL = np.r_[PASS_FREQ, STOP_FREQ]


def apply_group_scaling(net, factors, original):
    # Keep symmetric pairs exactly equal; avoid reparsing network for each trial.
    group = {'C1': 0, 'C5': 0, 'L2': 1, 'L4': 1, 'C3': 2}
    for el in net.elements:
        name = el['name']
        if name in group:
            param, unit = NAMES[name]
            el['params'][param] = f'{original[name] * float(factors[group[name]]) / unit:.12g}' + ('p' if param=='C' else 'n')


def sweep_loss(net, freqs):
    values=[]
    for f in freqs:
        s = solve_netlist_at_freq(net, float(f), method='matrix')
        if s.shape != (2,2):
            raise ValueError('Expect a two-port filter')
        values.append(-20.0 * np.log10(max(abs(s[1,0]), 1e-14)))
    return np.asarray(values)


def score_from_loss(loss):
    p = loss[:len(PASS_FREQ)]
    s = loss[len(PASS_FREQ):]
    # squared violations, in dB; passband ripple additionally gently controlled.
    violation_p=np.maximum(p-PASS_MAX_DB,0)
    violation_s=np.maximum(STOP_MIN_DB-s,0)
    return float(12*np.mean(violation_p**2) + np.mean(violation_s**2)
                 + .015*np.mean(p**2))


def main():
    if not NETFILE.exists():
        raise FileNotFoundError(NETFILE)
    net=read_netlist(str(NETFILE))
    elmap={el['name']:el for el in net.elements}
    missing=set(NAMES)-set(elmap)
    if missing: raise ValueError(f'Missing required components: {missing}')
    from components import to_si
    original={name:to_si(elmap[name]['params'][param]) for name,(param,_) in NAMES.items()}
    if not (np.isclose(original['C1'], original['C5']) and np.isclose(original['L2'],original['L4'])):
        raise ValueError('Original LC ladder must be symmetric')
    baseline=sweep_loss(net,F_EVAL)
    baseline_score=score_from_loss(baseline)
    counts=[0]
    def objective(x):
        counts[0]+=1
        apply_group_scaling(net,x,original)
        try:
            return score_from_loss(sweep_loss(net,F_EVAL))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return 1e12
    result=differential_evolution(objective,bounds=[(.65,1.45)]*3,
                    seed=7, maxiter=18, popsize=6, tol=1e-5, polish=True)
    best=np.asarray(result.x)
    apply_group_scaling(net,best,original)
    optimized=sweep_loss(net,F_EVAL)
    optimized_score=score_from_loss(optimized)
    # Guarantee no regression relative to baseline.
    if optimized_score > baseline_score:
        best=np.ones(3)
        apply_group_scaling(net,best,original)
        optimized=baseline.copy()
        optimized_score=baseline_score
    OUT.mkdir(exist_ok=True)
    output_net=OUT/'filter_lc_optimized.net'
    # Preserve existing text, comments, connectivity; only parameter tokens change.
    import re
    original_text=NETFILE.read_text(encoding='utf-8')
    replacements={name:elmap[name]['params'][NAMES[name][0]] for name in NAMES}
    newlines=[]
    for line in original_text.splitlines(keepends=True):
        match=re.match(r'^(\s*(?:CAP|IND)\s+(\w+)\s+.*?\s+)([CL])=([^\s;#!]+)(.*)$',line,re.I)
        if match and match.group(2) in replacements:
            line=match.group(1)+match.group(3)+'='+replacements[match.group(2)]+match.group(5)+'\n'
        newlines.append(line)
    output_net.write_text(''.join(newlines),encoding='utf-8')
    verify=read_netlist(str(output_net))
    if not np.allclose(sweep_loss(verify,F_EVAL),optimized,rtol=1e-8,atol=1e-7):
        raise AssertionError('Exported optimized netlist does not reproduce optimizer result')
    chartfreq=np.linspace(.05e9,2.0e9,181)
    base_net=read_netlist(str(NETFILE))
    d0=sweep_loss(base_net,chartfreq)
    d1=sweep_loss(verify,chartfreq)
    with (OUT/'filter_lc_before_after.csv').open('w',newline='') as fh:
        writer=csv.writer(fh)
        writer.writerow(['frequency_GHz','insertion_loss_before_dB','insertion_loss_after_dB'])
        for freq,a,b in zip(chartfreq,d0,d1): writer.writerow([f'{freq/1e9:.9g}',f'{a:.12g}',f'{b:.12g}'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(chartfreq/1e9,d0,label='Before optimization')
    ax.plot(chartfreq/1e9,d1,label='After optimization')
    ax.axhline(PASS_MAX_DB,ls=':',label='Passband limit 0.35 dB')
    ax.axhline(STOP_MIN_DB,ls=':',label='Stopband target 22 dB')
    ax.axvspan(.1,.75,alpha=.07)
    ax.axvspan(1.2,1.6,alpha=.07)
    ax.set(xlabel='Frequency (GHz)',ylabel='Insertion loss (dB)',title='LC filter optimization | team S-matrix solver')
    ax.set_ylim(-1,55); ax.grid(alpha=.3);ax.legend();fig.tight_layout()
    fig.savefig(OUT/'filter_lc_before_after.png',dpi=170);plt.close(fig)
    print('=== LC FILTER OPTIMIZATION (TEAM SOLVER) ===')
    print(f'Design targets: passband 0.1-0.75 GHz <= {PASS_MAX_DB} dB; stopband 1.2-1.6 GHz >= {STOP_MIN_DB} dB')
    print(f'Objective before: {baseline_score:.6f}; after: {optimized_score:.6f}')
    print('Scaled parameter groups [C1=C5, L2=L4, C3]:',np.array2string(best,precision=6))
    for name in NAMES:
        print(f'  {name}: {original[name]:.8g} SI  ->  {float(elmap[name]["params"][NAMES[name][0]][:-1])*NAMES[name][1]:.8g} SI')
    print(f'Passband peak loss: {baseline[:len(PASS_FREQ)].max():.4f} -> {optimized[:len(PASS_FREQ)].max():.4f} dB')
    print(f'Stopband minimum attenuation: {baseline[len(PASS_FREQ):].min():.4f} -> {optimized[len(PASS_FREQ):].min():.4f} dB')
    print(f'Function evaluations: {counts[0]}')
    print('PASS: exported optimized netlist reproduces optimized S21')
    print('Files:',output_net,OUT/'filter_lc_before_after.png',OUT/'filter_lc_before_after.csv',sep='\n  ')

if __name__ == '__main__': main()
