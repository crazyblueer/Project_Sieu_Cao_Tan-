"""Independent stochastic check of the FULL active amplifier's noise propagation.

Run: python3 test_noise_amplifier_montecarlo.py
Uses same assumed FET noise parameters as noise_active.py. Monte-Carlo directly
solves the interconnection for randomly generated independent component noise waves;
it does NOT use analyze_amplifier's covariance propagation in the simulation path.
It is a separate numerical check, not external ADS/AWR validation.
"""
import os
import numpy as np
from scipy.constants import Boltzmann as KB
from netlist import read_netlist
from engine import block_diag
from noise_analysis import passive_noise_covariance
from noise_active import EXAMPLE_NOISE, noise_params_to_c, analyze_amplifier

ROOT = os.path.dirname(os.path.abspath(__file__))
T0 = 290.0


def stochastic_full_network_check(net, freq_hz=2.4e9, trials=80000, seed=2026):
    blocks, connections, ext, _, ext_z0 = net.build_network(freq_hz, 50.)
    if len(ext) != 2 or any(abs(z-50)>1e-9 for z in ext_z0):
        raise ValueError('Requires two external ports and real 50-ohm reference')
    s_all, labels = block_diag(blocks)
    component_cs = []
    for s, pins in blocks:
        name = pins[0].rsplit('_p', 1)[0]
        is_transistor = any(e['name'] == name and e['type'] == 'S2P' for e in net.elements)
        component_cs.append(noise_params_to_c(s, **EXAMPLE_NOISE) if is_transistor
                            else passive_noise_covariance(s, T0))
    n = len(labels)
    cs = np.zeros((n, n), complex)
    cursor = 0
    for c in component_cs:
        m = c.shape[0]
        cs[cursor:cursor+m, cursor:cursor+m] = c
        cursor += m

    ext_set = set(ext)
    attached = {}
    internal_pairs = []
    for p, q in connections:
        if p in ext_set:
            attached[p] = q
        elif q in ext_set:
            attached[q] = p
        else:
            internal_pairs.append((labels.index(p),labels.index(q)))
    inside = sorted(set(i for pair in internal_pairs for i in pair))
    outside = [i for i in range(n) if i not in inside]
    ix = {i:j for j,i in enumerate(inside)}
    gamma = np.zeros((len(inside),len(inside)),complex)
    for i,j in internal_pairs:
        gamma[ix[i],ix[j]]=1
        gamma[ix[j],ix[i]]=1

    si = s_all[np.ix_(inside,inside)]
    se = s_all[np.ix_(outside,inside)]
    # The direct field-solve is deliberately separate from the covariance
    # transfer matrix used in noise_active.analyze_amplifier.
    rng = np.random.default_rng(seed)
    # Draw independent proper-complex noise waves for each component block.
    noise = np.zeros((n, trials), complex)
    cursor = 0
    for c in component_cs:
        m = c.shape[0]
        eig, u = np.linalg.eigh((c+c.conj().T)/(2*KB*T0))
        if eig.min() < -1e-8:
            raise ValueError('Component noise covariance is not PSD')
        root = (u*np.sqrt(np.clip(eig,0,None)))@u.conj().T
        z=(rng.standard_normal((m,trials))+1j*rng.standard_normal((m,trials)))/np.sqrt(2)
        noise[cursor:cursor+m,:] = root@z
        cursor += m
    ce = noise[outside]
    ci = noise[inside]
    # b_i = S_ii Gamma b_i + c_i for zero external incident waves
    bi = np.linalg.solve(np.eye(len(inside))-si@gamma,ci)
    be = se@gamma@bi+ce
    out_row = outside.index(labels.index(attached[ext[1]]))
    sample_c22 = KB*T0*np.mean(np.abs(be[out_row,:])**2)
    analytic = analyze_amplifier(net, freq_hz, {'Q1': EXAMPLE_NOISE})
    predicted_c22 = float(np.real(analytic['C'][1,1]))
    rel = abs(sample_c22/predicted_c22-1)
    print(f'Amplifier at {freq_hz/1e9:g} GHz; {trials} stochastic trials')
    print('Predicted C22 = %.9e W/Hz' % predicted_c22)
    print('Monte-Carlo C22 = %.9e W/Hz' % sample_c22)
    print('Relative discrepancy = %.3f %%' % (100*rel))
    print('NF from S/C model = %.6f dB' % analytic['NF_dB'])
    if rel >= .035:
        raise AssertionError('Monte-Carlo noise covariance differs by >= 3.5%')
    print('PASS: independent stochastic full-network noise propagation check')


if __name__=='__main__':
    net = read_netlist(os.path.join(ROOT,'amplifier.net'))
    print('WARNING: assumed transistor noise model, not real transistor data')
    stochastic_full_network_check(net)
