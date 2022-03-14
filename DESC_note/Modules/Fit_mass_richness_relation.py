import emcee, corner
import pickle
import sys
import numpy as np
import astropy.units as u
from astropy.table import Table, QTable
import iminuit
from iminuit import Minuit

def logM_richness_z(richness, z, logM_0, G, F):
    return logM_0 + G*np.log10((1+z)/(1 + 0.67)) + F*np.log10((richness/36))

def fit_M_lambda(file = 1, logm = '', logm_err = '',
                 z = '', z_err = '',
                 richness = '', richness_err = '',
                 abundance = '',
                 z_min = 1, z_max = 3,
                richness_min = 2, richness_max = 2,
                abundance_min = 2):

    m = {'M_0' : None, 'G' : None, 'F' : None}
    params = {'z_min' : z_min, 'z_max' : z_max, 'richness_min' : richness_min, 'z_min' : z_min, 'richness_max' : richness_max}
    mask = (file[z] >= z_min)*(file[z] < z_max)*(file[richness] > richness_min) * (file[richness] <= richness_max)*(file[abundance] > abundance_min)
    f_cut = file[mask]

    def chi2(logM_0, G, F): # -2ln(L)
        logMf = f_cut[logm]
        Redshift = f_cut[z]
        err_logM = f_cut[logm_err]
        Richness = f_cut[richness]
        return np.sum( ((logMf - logM_richness_z(Richness, Redshift, logM_0, G, F))/err_logM)**2 )
        
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
    
    def log_prior( logM_0 , G , F ):
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
    
    nwalkers, nstep, ndim = 2000, 150, 3
    initial = np.zeros((nwalkers, ndim))
    
    for i in range(nwalkers): 
        
        logM, G, F = minuit.values['logM_0'], minuit.values['G'], minuit.values['F']
        err_logM = minuit.errors['logM_0']
        err_G = minuit.errors['G']
        err_F = minuit.errors['F']
    
        initial[i,:]  = [logM + err_logM * np.random.randn(), G + err_G * np.random.randn(), F + err_F * np.random.randn()]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    sampler.run_mcmc(initial, nstep, progress=True)
    sampler_emcee = sampler.get_chain()
    return sampler_emcee, fit_minuit, params

