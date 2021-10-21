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
    
    def ___init___(self):
        
        self.name = 'Likelihood for cluster count Cosmology'
        
    def set_cosmological_definitions(self, cosmo = 1):
        
        self.cosmo = cosmo
        
        self.massdef = ccl.halos.massdef.MassDef('vir','critical', c_m_relation=None)
        
        self.hmd = ccl.halos.hmfunc.MassFuncDespali16(self.cosmo, mass_def=self.massdef)

    def dndlog10M(self, log10M, z):
        
        r"""
            returns the halo mass function
        """

        hmf = self.hmd.get_mass_function(self.cosmo, 10**np.array(log10M), 1./(1. + z))

        return hmf

    def dVdzdOmega(self,z):
    
        r"""
            redshift_solid_angle partial derivative comobile voulume
        """
    
        a = 1./(1. + z)
        
        da = ccl.background.angular_diameter_distance(self.cosmo, a) # Mpc
        
        ez = ccl.background.h_over_h0(self.cosmo, a) 
        
        dh = ccl.physical_constants.CLIGHT_HMPC / self.cosmo['h'] # Mpc
        
        return dh * da * da/( ez * a **2)

    def compute_tabulated_integrand_true_mass(self, z_middle = 1, logm_middle = 1):
        
        r"""
        returns the sky_area * hmf * d^2V/dzdOmega on a regular z-logm grid (the integrand for the cluster abundance for mass as a proxy)
        """

        integrand = np.zeros([len(logm_middle), len(z_middle)])

        for i, z_value in enumerate(z_middle):

            integrand[:,i] = self.sky_area * self.dndlog10M(logm_middle,z_value) * self.dVdzdOmega(z_value) 

        return integrand
    
    def compute_tabulated_integrand_richness(self, z_middle = 1, logm_middle = 1, lnlambda_bin = [], richness_model = 1):
        
        r"""
        returns the sky_area * hmf * dV/dzdOmega on a regular z-logm grid (the integrand for the cluster abundance for mass as a proxy)
        """

        integrand = np.zeros([len(logm_middle), len(z_middle)])

        for i, z_value in enumerate(z_middle):

            integrand[:,i] = self.richness.integral_logGaussian(lnlambda_bin, z_value, logm_middle)

        return integrand
    
    def Binned_predicted_cluster_abundance_proxy_mass(self, Redshift_bin = [], Proxy_bin = [], redshift_integrand_axis = [], proxy_integrand_axis = []):
        
        r"""
        returns the predicted number count in mass-z bins
        """
        
        N_th_matrix = np.zeros([len(Redshift_bin), len(Proxy_bin)])

        grid_tabulated = self.compute_tabulated_integrand_true_mass(z_middle = redshift_integrand_axis, logm_middle = proxy_integrand_axis)
        
        index_proxy = np.arange(len(proxy_integrand_axis))
        
        index_z = np.arange(len(redshift_integrand_axis))
        
        for i, proxy_bin in enumerate(Proxy_bin):

            mask_proxy = (proxy_integrand_axis >= proxy_bin[0])*(proxy_integrand_axis <= proxy_bin[1])

            proxy_cut = proxy_integrand_axis[mask_proxy]

            index_proxy_cut = index_proxy[mask_proxy]
            
            proxy_cut[0], proxy_cut[-1] = proxy_bin[0], proxy_bin[1]

            for j, z_bin in enumerate(Redshift_bin):
                
                z_down, z_up = z_bin[0], z_bin[1]

                mask_z = (redshift_integrand_axis >= z_bin[0])*(redshift_integrand_axis <= z_bin[1])
                
                z_cut = redshift_integrand_axis[mask_z]
    
                index_z_cut = index_z[mask_z]

                z_cut[0], z_cut[-1] = z_down, z_up
                
                grid_tabulated_cut = np.array([grid_tabulated[:,k][mask_proxy] for k in index_z_cut])

                N_th = simps(simps(grid_tabulated_cut, proxy_cut), z_cut)
                
                N_th_matrix[j,i] = N_th
                
        return N_th_matrix
    
    def Binned_predicted_cluster_abundance_proxy_richness(self, Redshift_bin = [], Proxy_bin = [], redshift_integrand_axis = [], proxy_integrand_axis = []):
        
        r"""
        returns the predicted number count in mass-z bins
        """
        N_th_matrix = np.zeros([len(Redshift_bin), len(Proxy_bin)])

        grid_tabulated = self.compute_tabulated_integrand_true_mass(z_middle = redshift_integrand_axis, logm_middle = proxy_integrand_axis)
        
        index_proxy = np.arange(len(proxy_integrand_axis))
        
        index_z = np.arange(len(redshift_integrand_axis))
        
        for i, proxy_bin in enumerate(Proxy_bin):

            for j, z_bin in enumerate(Redshift_bin):
                
                z_down, z_up = z_bin[0], z_bin[1]

                mask_z = (redshift_integrand_axis >= z_bin[0])*(redshift_integrand_axis <= z_bin[1])
                
                z_cut = redshift_integrand_axis[mask_z]
    
                index_z_cut = index_z[mask_z]

                z_cut[0], z_cut[-1] = z_down, z_up
                
                grid_tabulated_cut = np.array([grid_tabulated[:,k] for k in index_z_cut])
                
                grid_tabulated_cut_new = grid_tabulated_cut * self.compute_tabulated_integrand_richness(z_middle = z_cut, logm_middle = proxy_integrand_axis, lnlambda_bin = proxy_bin).T

                N_th = simps(simps(grid_tabulated_cut_new, proxy_integrand_axis), z_cut)
                
                N_th_matrix[j,i] = N_th
                
        return N_th_matrix

    def lnL_binned_Gaussian(self, N_th_matrix, N_obs_matrix, covariance_matrix):
        
        r"""
        returns the value of the log-likelihood for gaussian binned approach
        """

        delta = (N_obs_matrix - N_th_matrix).flatten()
        
        inv_covariance_matrix = np.linalg.inv((covariance_matrix))
                
        lnL_Gaussian = -0.5*np.sum(delta*inv_covariance_matrix.dot(delta)) 
        
        #lnL_Poissonian = np.sum(data_vector.flatten() * np.log(N_th_matrix.flatten()) - N_th_matrix.flatten())
        
        self.lnL_Gaussian = lnL_Gaussian

        #self.lnL_Poissonian = lnL_Poissonian

    def lnL_unbinned(self, z_middle, logm_middle, data):

        grid_mcmc = compute_hmf_plan(z_mid, logm_mid, cosmo)

        N_tot_th = simps(simps(grid_mcmc, z_mid), logm_mid)*(4*np.pi)*(1./.4)

        f = interpolate.RectBivariateSpline(z_mid, logm_mid, grid_mcmc)

        hmf_th_per_halo = f(redshift, np.log10(M200c), grid = False)

        return -(np.sum(np.log(hmf_th_per_halo)) - N_tot_th)

