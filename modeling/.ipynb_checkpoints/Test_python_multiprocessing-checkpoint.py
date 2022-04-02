import numpy as np
from scipy.integrate import simps
import time
from tqdm.auto import tqdm, trange
import pyccl as ccl
import multiprocessing
from math import gamma
import DATAOPS_Importance_Sampling as imp_sampling
import CL_COUNT_class_cluster_abundance as cl_count
from CL_COUNT_Sij_FLacasa import Sij_FLacasa
def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]

n = 100
Om = np.random.random(n)*(.4-.3)+.3
s8 = np.random.random(n)*(.9-.7)+.7

clc = cl_count.ClusterAbundance()
clc.sky_area = (0.25)*4*np.pi
clc.f_sky = clc.sky_area/4*np.pi

z_corner = np.linspace(0.25, 1.25, 6)
logm_corner = np.linspace(14, 14.8, 6)
Z_bin = binning(z_corner)
logMass_bin = binning(logm_corner)

z_grid = np.linspace(0., 3, 900)
logm_grid = np.linspace(12,16, 900)

pos = np.array([Om, s8]).T

def model(theta):
    Om_v, s8_v = theta
    cosmo_new = ccl.Cosmology(Omega_c = Om_v - 0.048254, Omega_b = 0.048254, h = 0.677, sigma8 = s8_v, n_s=0.96)
    massdef = ccl.halos.massdef.MassDef('vir', 'critical', c_m_relation=None)
    hmd = ccl.halos.hmfunc.MassFuncDespali16(cosmo_new, mass_def=massdef)
    clc.set_cosmology(cosmo = cosmo_new, hmd = hmd, massdef = massdef)
    clc.compute_multiplicity_grid_MZ(z_grid = z_grid, logm_grid = logm_grid)
    return clc.Cluster_Abundance_MZ(Redshift_bin = Z_bin, Proxy_bin = logMass_bin, method = 'simps')

def fct_n(n): return model(pos[n])

ti = time.time()
res_mp = imp_sampling.map(fct_n, np.arange(n), ordered=True)
tf = time.time()
print(tf-ti)
print(np.array(res_mp).shape)

res = []
ti = time.time()
for n_ in np.arange(n): res.append(fct_n(n_))
tf = time.time()
print(tf-ti)
print(np.array(res).shape)
