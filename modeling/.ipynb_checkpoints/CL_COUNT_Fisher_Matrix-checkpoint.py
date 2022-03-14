import numpy as np
import sys
import pyccl as ccl
from scipy.integrate import quad,simps, dblquad
sys.path.append('/pbs/throng/lsst/users/cpayerne/GitForThesis/DC2Analysis/modeling/')
import CL_COUNT_class_cluster_abundance as cl_count
import CL_COUNT_MVP_cluster_abundance_prediction as mvp
import DATAOPS_Fisher_matrix as fisher

class Fisher_Matrix():
    r"""Fisher Matrix for cluster abundance cosmology
    a. Binned cluster abundance (Gaussian, Poissonain, MVP)
    b. Unbinned cluster abundance 
    """
    def __init__(self, cosmo_true):
        r"""Create ClusterAbundance object"""
        self.forecast = fisher.Forecast()
        self.cosmo_true = cosmo_true
        self.True_value_0m_s8 = [self.cosmo_true['Omega_c'] + self.cosmo_true['Omega_b'], self.cosmo_true['sigma8']]
        
    def set_ClusterAbundance_Object(self, f_sky = 1):
        r"""Create ClusterAbundance object"""
        CA = cl_count.ClusterAbundance()
        CA.f_sky = f_sky
        CA.sky_area = CA.f_sky*4*np.pi
        self.z_min, self.z_max = 0.01, 3
        self.logm_min, self.logm_max = 14, 16
        z_grid = np.linspace(.001, 3, 1000)
        logm_grid = np.linspace(13.5,16.5, 1000)
        CA.z_grid = z_grid
        CA.logm_grid = logm_grid
        self.CA = CA
        
    def change_cosmo(self, Omega_m_new, sigma8_new):
        "returns new cosmology object with new values"""
        Omega_b = self.cosmo_true['Omega_b']
        cosmo_new = ccl.Cosmology(Omega_c = Omega_m_new - Omega_b, Omega_b = Omega_b, h = self.cosmo_true['h'], sigma8 = sigma8_new, n_s=self.cosmo_true['n_s'])
        return cosmo_new
        
    def set_cosmo_definitions(self, cosmo_new):
        r"""update cosmological definitions (halo mass function, halo bais, etc.) to the new cosmology"""
        massdef_new = ccl.halos.massdef.MassDef('vir', 'critical', c_m_relation=None)
        hmd_new = ccl.halos.MassFuncDespali16(cosmo_new, mass_def=massdef_new)
        self.CA.halobais = ccl.halos.hbias.HaloBiasTinker10(cosmo_new, mass_def= massdef_new, mass_def_strict=True)
        self.CA.set_cosmology(cosmo = cosmo_new, hmd = hmd_new, massdef = massdef_new)
    
    def compute_grid(self, bais = False):
        r"""compute grids for cluster abundance"""
        self.CA.compute_multiplicity_grid_MZ(z_grid = self.CA.z_grid, logm_grid = self.CA.logm_grid)
        if bais == True:
            self.CA.compute_halo_bias_grid_MZ(z_grid = self.CA.z_grid, logm_grid = self.CA.logm_grid, halobiais =self.CA.halobais)
        
    def model_Binned_Nth(self, Z_bin, logMass_bin, cosmo_new):
        r"""Model for binned Cluster Abundance as a function of cosmo"""
        self.set_cosmo_definitions(cosmo_new)
        self.compute_grid(bais = False)
        N_th_new = self.CA.Cluster_Abundance_MZ(Redshift_bin = Z_bin, Proxy_bin = logMass_bin, method = 'simps')
        return N_th_new

    def Fisher_matrix_binned(self, Z_bin, logMass_bin, cosmo, Abundance_true, cov_SSC, cov_Shot_Noise):
        r"""predict the fisher matrix for binned approach"""
        def model_Nth(theta):
            r"""define Binned Cluster abundance as a function of theta = Om, s8"""
            Omega_m_new, sigma8_new = theta
            cosmo_new = self.change_cosmo(Omega_m_new, sigma8_new)
            return self.model_Binned_Nth(Z_bin, logMass_bin, cosmo_new)
        
        def model_Nth_flatten(theta):
            return model_Nth(theta).flatten()
        
        #build covariance matrices: Gaussian, Poissonian, Gaussian(diagonal)
        Fisher_Binned_Poissonian = np.zeros([2, 2])
        Fisher_Binned_Gaussian_diag = np.zeros([2, 2])
        Fisher_Binned_Gaussian = np.zeros([2, 2])
        #compute second derivatives of binned cluster abundance
        dd = self.forecast.first_derivative(self.True_value_0m_s8, model_Nth_flatten, Abundance_true.shape, delta = 1e-5)
        #compute Fisher matrix
        for i in range(2):
            for j in range(2):
                Fisher_Binned_Poissonian[i,j] = np.sum(dd[i] * np.linalg.inv(cov_Shot_Noise).dot(dd[j]))
                Fisher_Binned_Gaussian_diag[i,j] = np.sum(dd[i] * np.linalg.inv(np.diag(np.diag(cov_Shot_Noise + cov_SSC))).dot(dd[j]))
                Fisher_Binned_Gaussian[i,j] = np.sum(dd[i] * np.linalg.inv(cov_Shot_Noise + cov_SSC).dot(dd[j]))
        cov_param_Binned_Poissonian = np.linalg.inv(Fisher_Binned_Poissonian)
        cov_param_Binned_Gaussian_diag = np.linalg.inv(Fisher_Binned_Gaussian_diag)
        cov_param_Binned_Gaussian = np.linalg.inv(Fisher_Binned_Gaussian)
        return cov_param_Binned_Poissonian, cov_param_Binned_Gaussian_diag, cov_param_Binned_Gaussian

    def Fisher_matrix_unbinned_Poissonian(self, Z_bin, logMass_bin, cosmo):
        r"""Fisher matrix for unbinned poissonian"""
        z_min, z_max = Z_bin
        logm_min, logm_max = logMass_bin

        def model_N_th_tot(theta):
            r"""Binned abundance prediction for a single bin"""
            Omega_m_new, sigma8_new = theta
            cosmo_new = self.change_cosmo(Omega_m_new, sigma8_new)
            return self.model_Binned_Nth([[z_min, z_max]], [[logm_min, logm_max]], cosmo_new)[0][0]

        def model_grid_ln_multiplicity(theta):
            r"""compute ln_multiplicity grid as a function of theta"""
            Omega_m_new, sigma8_new = theta
            cosmo_new = self.change_cosmo(Omega_m_new, sigma8_new)
            self.set_cosmo_definitions(cosmo_new)
            self.compute_grid(bais = False)
            self.CA.compute_multiplicity_grid_MZ(z_grid = self.CA.z_grid, logm_grid = self.CA.logm_grid)
            return np.log(self.CA.sky_area * self.CA.dN_dzdlogMdOmega)

        def av_2nd_derivative_ln_multiplicity(theta, model, delta = 1e-5):
            r"""average the second derivative of ln_multiplicity grid"""
            model_true = np.exp(model(theta))
            pdf = model_true/N_th_cosmo_true_unbinned
            index_z_grid = np.arange(len(self.CA.z_grid))
            index_logm_grid = np.arange(len(self.CA.logm_grid))
            mask_z = (self.CA.z_grid > z_min)*(self.CA.z_grid < z_max)
            mask_logm = (self.CA.logm_grid > logm_min)*(self.CA.logm_grid < logm_max)
            index_z_mask = index_z_grid[mask_z]
            index_logm_mask = index_logm_grid[mask_logm]
            res = np.zeros([len(theta),len(theta)])
            sec_derivative = self.forecast.second_derivative(self.True_value_0m_s8, model_grid_ln_multiplicity, model_true.shape)
            for i in range(len(theta)):
                for j in range(len(theta)):
                    if i >= j:
                        integrand = sec_derivative[i,j] * pdf
                        integrand_cut = np.array([integrand[:,i][mask_logm] for i in index_z_mask])
                        res[i,j] = simps(simps(integrand_cut, self.CA.logm_grid[mask_logm]), self.CA.z_grid[mask_z])
                        res[j,i] = res[i,j]
            return  res
        #total number of clusters
        N_th_cosmo_true_unbinned = model_N_th_tot(self.True_value_0m_s8)
        #second derivative of total abundance
        Ntot_second_derivative = self.forecast.second_derivative(self.True_value_0m_s8, model_N_th_tot, N_th_cosmo_true_unbinned.shape)
        #average of the second derivative of ln multiplicity
        av_2nd_derivative_ln_lambda = av_2nd_derivative_ln_multiplicity(self.True_value_0m_s8, model_grid_ln_multiplicity, delta = 1e-5)
        Fisher_unBinned_Poissonian = Ntot_second_derivative - N_th_cosmo_true_unbinned * av_2nd_derivative_ln_lambda
        cov_param_unBinned_Poissonian = np.linalg.inv(Fisher_unBinned_Poissonian)
        return cov_param_unBinned_Poissonian

    def Fisher_matrix_Binned_MVP(self, Z_bin, logMass_bin, cosmo, cov_SSC):
        r"""Fisher matrix for Binned MVP"""
        cov_SSC_diag = cov_SSC.diagonal()
        N_th_true_cosmo = self.model_Binned_Nth(Z_bin, logMass_bin, cosmo)
        MVP = mvp.MVP(N_th_true_cosmo.flatten(), cov_SSC_diag)
        MVP._set_axis(5, N_th_true_cosmo.flatten(), cov_SSC_diag)
        def ln_P_mvp_grid(theta):
            r"""compute log(P) for each predicted cluster abundance"""
            Omega_m_new, sigma8_new = theta
            cosmo_new = self.change_cosmo(Omega_m_new, sigma8_new)
            self.set_cosmo_definitions(cosmo_new)
            N_th = self.model_Binned_Nth(Z_bin, logMass_bin, cosmo_new).flatten()
            N, P = MVP.p_mvp(N_th, cov_SSC_diag)
            P_grid = np.zeros([len(N_th_true_cosmo.flatten()), MVP.n_max])
            for i, n_th in enumerate(N_th):
                P_grid[i,:][N[i]] = P[i]/np.sum(P[i])
            return np.log(P_grid)
        #compute ln_P
        ln_P = ln_P_mvp_grid(self.True_value_0m_s8)
        #compute first derivative
        first_derivative_ln_P = self.forecast.first_derivative(self.True_value_0m_s8, ln_P_mvp_grid, ln_P.shape, delta = 1e-5)
        Fisher_matrix_MVP = np.zeros([2,2])
        for i in range(2):
            for j in range(2):
                if i >= j:
                    res = 0
                    for k in range(len(Z_bin)*len(logMass_bin)):
                        mask_finite = np.isfinite(first_derivative_ln_P[i][k])*np.isfinite(first_derivative_ln_P[j][k])
                        p = np.exp(ln_P[k][mask_finite])/np.sum(np.exp(ln_P[k][mask_finite]))
                        res = res + np.sum(p * first_derivative_ln_P[i][k][mask_finite]*first_derivative_ln_P[j][k][mask_finite])
                    Fisher_matrix_MVP[i,j] = res
                    Fisher_matrix_MVP[j,i] = res
        cov_param_MVP = np.linalg.inv(Fisher_matrix_MVP)
        return cov_param_MVP
    
    def Fisher_matrix_Binned_MVP_per_z_bin(self, Z_bin, logMass_bin, cosmo, cov_SSC):
        r"""Fisher matrix for Binned MVP"""
        cov_SSC_diag = cov_SSC.diagonal()
        N_th_true_cosmo = self.model_Binned_Nth(Z_bin, logMass_bin, cosmo)
        MVP = mvp.MVP(N_th_true_cosmo.flatten(), cov_SSC_diag)
        MVP._set_axis(5, N_th_true_cosmo.flatten(), cov_SSC_diag)
        def ln_P_mvp_grid(theta):
            r"""compute log(P) for each predicted cluster abundance"""
            Omega_m_new, sigma8_new = theta
            cosmo_new = self.change_cosmo(Omega_m_new, sigma8_new)
            self.set_cosmo_definitions(cosmo_new)
            N_th = self.model_Binned_Nth(Z_bin, logMass_bin, cosmo_new).flatten()
            N, P = MVP.p_mvp(N_th, cov_SSC_diag)
            P_grid = np.zeros([len(N_th_true_cosmo.flatten()), MVP.n_max])
            for i, n_th in enumerate(N_th):
                P_grid[i,:][N[i]] = P[i]/np.sum(P[i])
            return np.log(P_grid)
        #compute ln_P
        ln_P = ln_P_mvp_grid(self.True_value_0m_s8)
        #compute first derivative
        first_derivative_ln_P = self.forecast.first_derivative(self.True_value_0m_s8, ln_P_mvp_grid, ln_P.shape, delta = 1e-5)
        Fisher_matrix_MVP = np.zeros([2,2])
        for i in range(2):
            for j in range(2):
                if i >= j:
                    res = 0
                    for k in range(len(Z_bin)*len(logMass_bin)):
                        mask_finite = np.isfinite(first_derivative_ln_P[i][k])*np.isfinite(first_derivative_ln_P[j][k])
                        p = np.exp(ln_P[k][mask_finite])/np.sum(np.exp(ln_P[k][mask_finite]))
                        res = res + np.sum(p * first_derivative_ln_P[i][k][mask_finite]*first_derivative_ln_P[j][k][mask_finite])
                    Fisher_matrix_MVP[i,j] = res
                    Fisher_matrix_MVP[j,i] = res
        cov_param_MVP = np.linalg.inv(Fisher_matrix_MVP)
        return cov_param_MVP
