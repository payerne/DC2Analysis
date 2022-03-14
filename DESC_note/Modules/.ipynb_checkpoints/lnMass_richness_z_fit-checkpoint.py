import emcee, corner
import GCRCatalogs
import matplotlib.pyplot as plt
import pickle
import sys
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.table import Table, QTable
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_3d
cosmo_astropy = FlatLambdaCDM(H0=71.0, Om0=0.265, Ob0 = 0.0448)
import iminuit
from iminuit import Minuit


def load(filename, **kwargs):
    
    """Loads GalaxyCluster object to filename using Pickle"""
    
    with open(filename, 'rb') as fin:
        
        return pickle.load(fin, **kwargs)

cosmodc2_true = load('/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/p_RedMapper_clusters/paper_dc2_galaxy_cluster_mass/data_DS/Halo_mass/prf_cosmodc2_true_ellipticity_true_redshift_DK_fix.pkl')
cosmodc2_flex = load('/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/p_RedMapper_clusters/paper_dc2_galaxy_cluster_mass/data_DS/Halo_mass/prf_cosmodc2_true_ellipticity_flex_redshift_DK_fix.pkl')
cosmodc2_BPZ = load('/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/p_RedMapper_clusters/paper_dc2_galaxy_cluster_mass/data_DS/Halo_mass/prf_cosmodc2_true_ellipticity_BPZ_no_cut_redshift_DK_fix.pkl')
SkySim5000_RM = load('/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/p_RedMapper_clusters/paper_dc2_galaxy_cluster_mass/data_DS/Halo_mass/prf_SkySim5000_stack_RedMapper_target_cosmoDC2.pkl')
SkySim5000_cosmoDC2 = load('/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/p_RedMapper_clusters/paper_dc2_galaxy_cluster_mass/data_DS/Halo_mass/prf_SkySim50000_stack_cosmoDC2_target_RedMapper.pkl')
dc2HSM = load('/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/p_RedMapper_clusters/paper_dc2_galaxy_cluster_mass/data_DS/Halo_mass/prf_BPZ_z_HSM_shape.pkl')
dc2Metacal = load('/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/p_RedMapper_clusters/paper_dc2_galaxy_cluster_mass/data_DS/Halo_mass/prf_BPZ_z_Metacal_shape.pkl')

file_to_fit = [dc2Metacal]

file_label = ['dc2_Metacal.pkl']

def logM_richness_z(richness, z, logM_0, G, F):
    
    return logM_0 + G*np.log10((1+z)/(1 + 0.67)) + F*np.log10((richness/36))

z_min, z_max = 0.2, 1
richness_min, richness_max = 30, 500

for j, f in enumerate(file_to_fit):
    
    print(file_label[j])
    
    m = {'M_0' : 1, 'G' : 1, 'F' : 1}
    m_params = {'z_min' : z_min, 'z_max' : z_max, 'richness_min' : richness_min, 'z_min' : z_min, 'richness_max' : richness_max}
    
    mask = (f[0]['z_mean'] >= z_min)*(f[0]['z_mean'] < z_max)*(f[0]['richness'] > richness_min)*(f[0]['richness'] <= richness_max)#*(f[0]['n_stack'] > 50)
    f_cut = f[0][mask]

    def chi2(logM_0, G, F): # -2ln(L)
        
        logM = (f_cut['logm200'])
        z = f_cut['z_mean']
        err_logM = f_cut['logm200_err']
        richness =  f_cut['richness']
        
        return np.sum( ((logM - logM_richness_z(richness, z, logM_0, G, F))/err_logM)**2 )
        
    
    minuit = Minuit(chi2, logM_0 = 14, 
                   G = 0, F = 0,
                   limit_logM_0 = (12,16),limit_G = (-10,10), limit_F = (-10,10),
                   errordef = 1)
    
    minuit.migrad(),minuit.hesse(),minuit.minos()
    
    print(minuit.params)
    
    m = {'logM_0' : minuit.values['logM_0'], 
         'G' : minuit.values['G'], 
         'F' : minuit.values['F'],
         'logM_0_err' : minuit.errors['logM_0'], 
         'G_err' : minuit.errors['G'], 
         'F_err' : minuit.errors['F']}
    
    fit_minuit = m
    
    def log_prior(logM_0, G, F):
        
        if (logM_0 < 11) or (logM_0 > 16): 
            
            return - np.inf
        
        elif (G < - 10) or (G > 10): 
            
            return - np.inf
        
        elif (F < -10) or (F > 10): 
            
            return - np.inf
        
        return 0
    
    def log_probability(theta):
        
        logM_0, G, F = theta
        
        return -0.5*chi2(logM_0, G, F) + log_prior(logM_0, G, F)
    
    nwalkers, ndim = 2000, 3
    
    initial = np.zeros((nwalkers, ndim))
    
    def random(mean, width):
        down, up = mean-width/2, mean+width/2
        return np.random.random()*(up - down) + down
    
    for i in range(nwalkers): 
        
        logM, G, F = minuit.values['logM_0'], minuit.values['G'], minuit.values['F']
        err_logM = 0.5
        err_G = 0.3
        err_F = 0.3
        
        initial[i,:]  = [logM + err_logM * np.random.randn(), G + err_G * np.random.randn(), F + err_F * np.random.randn()]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    
    sampler.run_mcmc(initial, 700, progress=True)
    
    sampler_emcee = sampler.get_chain()
    
    T = Table()
    
    T[file_label[j]] = [sampler_emcee,fit_minuit,m_params]
    
    with open('/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/p_RedMapper_clusters/paper_dc2_galaxy_cluster_mass/data_DS/Halo_mass/M_lambda_constraint_' + file_label[j], 'wb') as file:
    
        pickle.dump(T, file)
