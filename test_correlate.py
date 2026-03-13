import numpy as np
import scipy.io as sio

a = sio.loadmat('data/meg/MEGRDMs_2D.mat')
meg = a['MEGRDMs_2D']  # (36, 36, 1201, 20)
meg_mean = meg.mean(axis=3)

spatial = sio.loadmat('data/meg/structureRDM_sq.mat')['RDM']['RDM'][0,0]

tri = np.triu_indices(36, k=1)
spatial_vec = spatial[tri]

corrs = []
from scipy.stats import spearmanr
for t in range(meg_mean.shape[2]):
    meg_vec = meg_mean[:,:,t][tri]
    rho, _ = spearmanr(spatial_vec, meg_vec)
    corrs.append(rho)

print(f"Max correlation: {np.max(corrs)} at time {(np.argmax(corrs) - 200)} ms")
