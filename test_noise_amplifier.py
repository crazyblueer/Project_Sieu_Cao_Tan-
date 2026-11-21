"""Run active amplifier noise sweep and optional independent scikit-rf check.

Install: /usr/local/bin/python3 -m pip install numpy scipy matplotlib scikit-rf
Run: /usr/local/bin/python3 test_noise_amplifier.py
NOTE: transistor noise parameters are illustrative, not measured/from fet.s2p.
"""
import os, csv
import numpy as np
from netlist import read_netlist
from noise_active import (EXAMPLE_NOISE, analyze_amplifier,
                          noise_params_to_c, nf_from_wave_covariance,
                          f_from_noise_params)

ROOT=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(ROOT,'results_noise')


def independently_check_transistor(net, f=2.4e9):
    # An analytical check first, plus a SEPARATE software-library check when present.
    elem=next(e for e in net.elements if e['type']=='S2P')
    s,_=net._element_block(elem,f,50.)
    c=noise_params_to_c(s, **EXAMPLE_NOISE)
    samples=[0, .3*np.exp(.4j), -.4j, EXAMPLE_NOISE['gamma_opt']]
    for gamma in samples:
        own=nf_from_wave_covariance(s,c,gamma)
        theoretical=f_from_noise_params(gamma,**EXAMPLE_NOISE)
        err=abs(own-theoretical)
        print(f'  gamma={gamma:.3f}: F(wave)={own:.9f}, F(analytic)={theoretical:.9f}, err={err:.2e}')
        assert err<1e-8
    try:
        import skrf as rf
    except ImportError:
        print('  scikit-rf UNAVAILABLE: independent-tool noise comparison NOT RUN.')
        return False
    ntwk=rf.Network(frequency=rf.Frequency.from_f([f], unit='hz'),s=s[None],z0=50.)
    ntwk.set_noise_a(noise_freq=ntwk.frequency, nfmin_db=EXAMPLE_NOISE['nfmin_db'],
                     gamma_opt=EXAMPLE_NOISE['gamma_opt'],rn=EXAMPLE_NOISE['rn_ohm'])
    for gamma in samples:
        # gamma=(Zs-Z0)/(Zs+Z0), positive-real Zs for |gamma|<1
        z_source=50*(1+gamma)/(1-gamma)
        independent=float(np.asarray(ntwk.nf(z_source)).ravel()[0])
        our=nf_from_wave_covariance(s,c,gamma)
        print(f'  scikit-rf: F={independent:.9f}, F(ours)={our:.9f}, err={abs(independent-our):.2e}')
        assert abs(independent-our)<1e-7
    print('PASS: scikit-rf transistor NF comparison')
    return True


def main():
    os.makedirs(OUT,exist_ok=True)
    net=read_netlist(os.path.join(ROOT,'amplifier.net'))
    print('WARNING: EXAMPLE transistor noise parameters are ASSUMED, NOT contained in fet.s2p.')
    print('Assumptions:',EXAMPLE_NOISE)
    validated=independently_check_transistor(net)
    freqs=np.linspace(1e9,4e9,301)
    rows=[]
    for f in freqs:
        r=analyze_amplifier(net,f,{'Q1':EXAMPLE_NOISE})
        s=r['S']; c=r['C']
        rows.append(dict(f_GHz=f/1e9,NF_dB=r['NF_dB'],
                         gain_S21_dB=20*np.log10(max(abs(s[1,0]),1e-30)),
                         C22_W_per_Hz=float(c[1,1].real)))
    at_center=analyze_amplifier(net,2.4e9,{'Q1':EXAMPLE_NOISE})
    print(f'Amplifier @ 2.4 GHz (hypothetical noise model): NF={at_center["NF_dB"]:.6f} dB, '
          f'|S21|={abs(at_center["S"][1,0]):.5f}')
    p=os.path.join(OUT,'amplifier_noise_assumed.csv')
    with open(p,'w',newline='',encoding='utf-8') as fp:
        w=csv.DictWriter(fp,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True)
    axs[0].plot([r['f_GHz'] for r in rows],[r['NF_dB'] for r in rows]);axs[0].set_ylabel('Noise figure (dB)')
    axs[1].plot([r['f_GHz'] for r in rows],[r['gain_S21_dB'] for r in rows]);axs[1].set_ylabel('20 log10 |S21| (dB)')
    axs[1].set_xlabel('Frequency (GHz)')
    for ax in axs: ax.grid(True,alpha=.3)
    fig.suptitle('Amplifier: assumed transistor noise parameters (illustrative, NOT measured)')
    fig.tight_layout();q=os.path.join(OUT,'amplifier_noise_assumed.png');fig.savefig(q,dpi=160);plt.close(fig)
    print('Frequency sweep PASS: 301 points; CSV:',p,'\nPlot:',q)
    if not validated: print('INDEPENDENT TOOL CHECK PENDING: install scikit-rf and rerun.')

if __name__=='__main__': main()
