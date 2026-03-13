import numpy as np
import scipy.io as sio

def load_mat(file_path):
    a = sio.loadmat(file_path)
    return a['RDM']['RDM'][0,0]

spatial_rdm = load_mat('data/meg/structureRDM_sq.mat')
semantic_rdm = load_mat('data/meg/semanticRDM_sq.mat')

print(f"Spatial RDM shape: {spatial_rdm.shape}")
print(f"Semantic RDM shape: {semantic_rdm.shape}")
