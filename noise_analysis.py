"""Thermal-noise extension for the Topic 7 S-matrix simulator.

Copy next to engine.py, netlist.py and components.py.
Scope: passive isothermal networks; real equal port reference z0 (default 50 ohm).
Internal correlated noise-wave PSD C is in W/Hz.
S2P components require an explicit noise model and are rejected intentionally.
"""
import numpy as np
from scipy.constants import Boltzmann as K_B
from engine import block_diag, connect_matrix

T0 = 290.0


def passive_noise_covariance(S, temperature=T0, tolerance=1e-9):
    """C = k_B*T*(I-S S^H). Rejects non-passive blocks."""
    S = np.asarray(S, dtype=complex)
    ident = np.eye(S.shape[0], dtype=complex)
    q = ident - S @ S.conj().T
    q = (q + q.conj().T) / 2
    eig = np.linalg.eigvalsh(q)
    if eig.min() < -tolerance:
        raise ValueError("Non-passive block: cannot infer noise from S alone")
    # Project tiny negative eigenvalues to 0 (floating-point roundoff).
    vals, vecs = np.linalg.eigh(q)
    q = (vecs * np.maximum(vals, 0)) @ vecs.conj().T
    return K_B * float(temperature) * q


def _block_diag_covariances(covs):
    n = sum(c.shape[0] for c in covs)
    out = np.zeros((n, n), dtype=complex)
    cursor = 0
    for c in covs:
        m = c.shape[0]
        out[cursor:cursor + m, cursor:cursor + m] = c
        cursor += m
    return out


def analyze_passive_noise(net, frequency, temperature=T0, z0=50.0):
    """Return external S, intrinsic noise correlation C, and noise figure (2 ports).

    Requires no S2P elements, equal real port references z0, and one temperature.
    Noise figure uses available-wave gain for a matched source and load.
    """
    if any(e['type'] == 'S2P' for e in net.elements):
        raise ValueError("S2P transistor/device requires explicit noise parameters or correlation matrix")
    if not np.isreal(z0) or z0 <= 0:
        raise ValueError("This first implementation requires positive real z0")
    blocks, connections, ext, pin_z0, ext_z0 = net.build_network(frequency, z0)
    if any(abs(complex(z) - z0) > 1e-12 for z in ext_z0):
        raise ValueError("This first implementation requires all external port Z0 = z0")
    full_s, labels = block_diag(blocks)
    full_c = _block_diag_covariances(
        [passive_noise_covariance(s, temperature) for s, _ in blocks]
    )
    ext_set = set(ext)
    pin_to_ext = {}
    pairs = []
    for x, y in connections:
        if x in ext_set:
            pin_to_ext[x] = y
        elif y in ext_set:
            pin_to_ext[y] = x
        else:
            pairs.append((labels.index(x), labels.index(y)))
    internal = sorted({v for pair in pairs for v in pair})
    outside = [i for i in range(len(labels)) if i not in internal]
    index = {p: j for j, p in enumerate(internal)}
    gamma = np.zeros((len(internal), len(internal)), dtype=complex)
    # Valid only for equal *real* reference impedances: each connection is a swap.
    for p, q in pairs:
        gamma[index[p], index[q]] = 1.0
        gamma[index[q], index[p]] = 1.0
    S_ei = full_s[np.ix_(outside, internal)]
    S_ii = full_s[np.ix_(internal, internal)]
    S_ie = full_s[np.ix_(internal, outside)]
    S_ee = full_s[np.ix_(outside, outside)]
    if internal:
        # A = S_ei * Gamma * inv(I-S_ii*Gamma); solve transposed for stability.
        h = np.linalg.solve((np.eye(len(internal)) - S_ii @ gamma).T,
                            (S_ei @ gamma).T).T
    else:
        h = np.zeros((len(outside), 0), dtype=complex)
    transform = np.concatenate([np.eye(len(outside)), h], axis=1)
    permutation = outside + internal
    c_perm = full_c[np.ix_(permutation, permutation)]
    s_out = S_ee + h @ S_ie
    c_out = transform @ c_perm @ transform.conj().T
    # Rearrange external ports in user PORT order.
    ordered = [pin_to_ext[label] for label in ext]
    pos = [outside.index(labels.index(pin)) for pin in ordered]
    s_out = s_out[np.ix_(pos, pos)]
    c_out = c_out[np.ix_(pos, pos)]
    nf_db = None
    f_linear = None
    if len(pos) == 2:
        gain = abs(s_out[1, 0])**2
        if gain > 1e-30:
            # Matched source at T0, matched/noiseless load; intrinsic noise c2.
            f_linear = 1 + float(c_out[1, 1].real) / (K_B * T0 * gain)
            nf_db = 10 * np.log10(f_linear)
    return {'S': s_out, 'C': c_out, 'NF_dB': nf_db, 'F': f_linear,
            'f_Hz': float(frequency), 'temperature_K': float(temperature)}


if __name__ == '__main__':
    from netlist import read_netlist
    import os
    root = os.path.dirname(__file__)
    for name, f in [('filter_lc.net', 0.5e9), ('wilkinson.net', 2.4e9),
                    ('branchline.net', 2.4e9)]:
        net = read_netlist(os.path.join(root, name))
        out = analyze_passive_noise(net, f)
        print(name, 'f=', f, 'NF(dB)=', out['NF_dB'],
              'C diag [W/Hz]=', np.diag(out['C']).real)
        equilibrium = K_B * T0 * (np.eye(out['S'].shape[0]) - out['S'] @ out['S'].conj().T)
        err = np.linalg.norm((out['C'] - equilibrium) / (K_B * T0))
        print('  isothermal equilibrium covariance residual:', err)
        assert err < 1e-8
