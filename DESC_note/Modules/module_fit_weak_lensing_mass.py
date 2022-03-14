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
cosmo_astropy.critical_density(0.4).to(u.Msun / u.Mpc**3).value
sys.path.append('/pbs/throng/lsst/users/cpayerne/GitForThesis/DC2Analysis')

import make_profile as prf

sys.path.append('/pbs/throng/lsst/users/cpayerne/GitForThesis/DC2Analysis/modeling/')

import miscentering as mis
import two_halo_term as twoh
import mass_conversion as utils
import clmm
import clmm.dataops
from clmm.dataops import compute_tangential_and_cross_components, make_radial_profile, make_bins
from clmm.galaxycluster import GalaxyCluster
import clmm.utils as u
from clmm import Cosmology
from clmm.support import mock_data as mock
import pyccl as ccl
cosmo = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)
cosmo_astropy = FlatLambdaCDM(H0=71.0, Om0=0.265, Ob0 = 0.0448)
cosmo_clmm = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)
cosmo_ccl  = ccl.Cosmology(Omega_c=0.265-0.0448, Omega_b=0.0448, h=0.71, A_s=2.1e-9, n_s=0.96, Neff=0, Omega_g=0)

moo = clmm.Modeling(massdef = 'critical', delta_mdef = 200, halo_profile_model = 'nfw')
moo.set_cosmo(cosmo_clmm)
deff = ccl.halos.massdef.MassDef(200, 'critical', c_m_relation=None)
conc = ccl.halos.concentration.ConcentrationDiemer15(mdef=deff)

def modele_ds(r, logm, c, cluster_z):

    m = 10.**logm 
    moo.set_mass(m), moo.set_concentration(c)
    deltasigma = []
    for i, xr in enumerate(r):
        deltasigma.append(moo.eval_excess_surface_density(xr, cluster_z))
    return np.array(deltasigma)

def fit(profile = 1, covariance = 1):
    
    fit_data_name = ['mask',
                     'chi2ndof', 
                     'logm200','logm200_err','logm200_err_forecast',
                     'c_w', 'c_w_err',
                     'richness','richness_err', 
                     '1h_term', '2h_term','radius_model']
    
    tab = {name : [] for name in fit_data_name}
    
    kk = np.logspace(-5,5 ,100000)

    for j, p in enumerate(profile):
        
        infos_cov = covariance[j]
        
        cluster_z, R, y_exp, cov_t = p['z_mean'], p['radius'], p['gt'], infos_cov['cov_t']

        Pk = twoh.compute_Pk(kk, cluster_z, cosmo_ccl)

        ds_unbaised = twoh.ds_two_halo_term_unbaised(R, cluster_z, cosmo_ccl, kk, Pk)
        
        rmin, rmax = (1.53 + 1*0.25)*cluster_z + (0.38 + 1*0.10), 5.5
        
        #rmin, rmax = 1.5, 20

        mask = (R > rmin)*(R <= rmax)
        
        logm = np.logspace(11,16,3000)
        
        c200array = conc._concentration(cosmo_ccl, logm, 1/(1 + cluster_z))
        
        def chi2(logm200, c200):

            c200 = np.interp(logm200, logm, c200array)

            bais = twoh.halo_bais(logm = logm200, concentration = c200,
                                         mdef = 'critical', Delta = 200, halo_def = 'nfw',
                                         cluster_z = cluster_z, cosmo_ccl = cosmo_ccl)

            y_predict = modele_ds(R, logm200, c200, cluster_z) #+ bais * ds_unbaised
            
            d = (y_predict - y_exp)

            d = np.array([d[i] if mask[i] == True else 0 for i in range(len(mask))])

            inv_cov = np.linalg.inv(np.diag(np.diag(cov_t)))
            
            #inv_cov = np.linalg.inv((cov_t))

            chi_2 = np.sum(d*inv_cov.dot(d))                                         

            return chi_2
        
        try:
            
            print('1')

            minuit = Minuit(chi2, logm200 = 14,c200 = 5, fix_c200 = True,
                       limit_logm200 = (11,16),limit_c200 = (0.01,20),
                       errordef = 1)

            minuit.migrad(),minuit.hesse(),minuit.minos()

            chi2 = minuit.fval/(len(mask[mask == True])-1)

            logm_fit = minuit.values['logm200']

            logm_fit_err = minuit.errors['logm200']

            c_fit = minuit.values['c200']

            c_fit_err = minuit.errors['c200']

            radius, y_exp, y_exp_err = R, y_exp, np.sqrt(cov_t.diagonal())

            ds_1h_term = modele_ds(R,logm_fit,c_fit,cluster_z)

            hbiais = twoh.halo_bais(logm = logm_fit, concentration = c_fit,
                                             mdef = 'critical', Delta = 200, halo_def = 'nfw',
                                             cluster_z = cluster_z, cosmo_ccl = cosmo_ccl)

            ds_2h_term = hbiais * ds_unbaised

            def ds(r, logm_value):
                c200 = np.interp(logm_value, logm, c200array)
                hbias = twoh.halo_bais(logm = logm_value, concentration = c200,
                                             mdef = 'critical', Delta = 200, halo_def = 'nfw',
                                             cluster_z = cluster_z, cosmo_ccl = cosmo_ccl)
                ds_2h = hbias*ds_unbaised

                return  modele_ds(r,logm_value,c200,cluster_z) + ds_2h

            def Forecast(logmv):
                dlogm = 0.001
                ds_logm = ds(R, logmv)
                ds_logm_dlogm = ds(R, logmv + dlogm)
                data = (ds_logm_dlogm - ds_logm)/dlogm
                data = np.array([data[i] if mask[i] == True else 0 for i in range(len(mask))])
                inv_cov = np.linalg.inv(np.diag(np.diag(cov_t)))
                err = 1/(np.sum(data*inv_cov.dot(data)))
                return np.sqrt(err)

            logm_fit_err_forecast = Forecast(logm_fit)

            dat_col = [mask, 
                       chi2, 
                       logm_fit,logm_fit_err, logm_fit_err_forecast, 
                       c_fit, c_fit_err, 
                       p['obs_mean'], p['obs_rms'],
                      ds_1h_term, ds_2h_term, R]

        except: 
            print('oups')
            
            dat_col = [-1 for i in range(len(fit_data_name))]
            
        for i, name in enumerate(fit_data_name):
            
            tab[name].append(dat_col[i])

    for name in fit_data_name:
        
        profile[name] = tab[name]
        
    mask = profile['c_w'] != - 1
        
    return profile[mask]
