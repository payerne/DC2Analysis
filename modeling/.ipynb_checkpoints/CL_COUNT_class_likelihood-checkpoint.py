import pandas as pd
import numpy as np
import pickle
import pyccl as ccl
import numpy as np
import healpy
import scipy, pickle
from scipy import stats
from scipy.integrate import quad,simps, dblquad
from scipy import interpolate
import pickle as pkl
import matplotlib.pyplot as plt
from itertools import combinations, chain
def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]
class Likelihood():
    r"""
        compute likelihood :
            a. for the binned gaussian case
            b. for the binned poissonian case
            c. for the un-binned poissonian case
    """
    def ___init___(self):
        self.name = 'Likelihood for cluster count Cosmology'

    def lnLikelihood_Binned_Gaussian(self, N_th_matrix, N_obs_matrix, covariance_matrix):
        r"""
        returns the value of the log-likelihood for gaussian binned approach
        """
        delta = (N_obs_matrix - N_th_matrix).flatten()
        inv_covariance_matrix = np.linalg.inv((covariance_matrix))
        lnL_Gaussian = -0.5*np.sum(delta*inv_covariance_matrix.dot(delta)) 
        self.lnL_Gaussian = lnL_Gaussian
        
    def lnLikelihood_Binned_Poissonian(self, N_th_matrix, N_obs_matrix):
        r"""
        returns the value of the log-likelihood for Poissonian binned approach
        """
        lnL_Poissonian = np.sum(N_obs_matrix.flatten() * np.log(N_th_matrix.flatten()) - N_th_matrix.flatten())
        self.lnL_Poissonian = lnL_Poissonian
        
    def lnLikelihood_UnBinned_Poissonian(self, dN_dzdlogMdOmega, N_tot):
        
        self.UnBinned_Poissonian = np.sum(np.log(dN_dzdlogMdOmega)) - N_tot