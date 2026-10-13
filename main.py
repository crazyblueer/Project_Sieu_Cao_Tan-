from components import tline_ideal
from components import renormalize, renormalize_general
import numpy as np

# (a) Tuong thich nguoc: Z0 thuc, bang nhau -> phai giong renormalize cu
S_old = tline_ideal(np.pi/2)
print(np.allclose(renormalize_general(S_old, 50*np.sqrt(2), 50),
                  renormalize(S_old, 50*np.sqrt(2), 50)))   # True

# (b) Sanity: doi ve chinh no -> khong doi; di va ve -> ve nguyen ban
z = [50+10j, 50+10j]
print(np.allclose(renormalize_general(S_old, z, z), S_old))                    # True
print(np.allclose(renormalize_general(renormalize_general(S_old, 50, z), z, 50), S_old))  # True

# (c) Cross-check scikit-rf voi power wave (s_def='power')
import skrf as rf
ntwk = rf.Network(frequency=rf.Frequency(2.4,2.4,1,'ghz'),
                  s=S_old.reshape(1,2,2), z0=50*np.sqrt(2))
ntwk.renormalize([50+10j, 50+10j], s_def='power')
print(np.round(ntwk.s[0],4))
print(np.round(renormalize_general(S_old, 50*np.sqrt(2), [50+10j,50+10j]),4))
# hai cai PHAI khop
