import sys
import os
os.environ['CLMM_MODELING_BACKEND'] = 'nc' # here you may choose ccl or nc (NumCosmo)
sys.path.append('/pbs/throng/lsst/users/cpayerne/CLMM/examples/support')
try: import clmm
except:
    import notebook_install
    notebook_install.install_clmm_pipeline(upgrade=False)
    import clmm
import matplotlib.pyplot as plt
plt.style.use('classic')
import numpy as np
from numpy import random

import numpy as np
from astropy.table import Table
import clmm.modeling as modeling
from scipy.integrate import quad
from astropy import units as u
from astropy import constants as const

class Signal():
    
    def __init__(self, cosmo = 1):
        
        self.cosmo = cosmo
        
        self.is_sigmoid = None
        
    def _set_halo_def(self, mass_def = 'mean', delta_mdef = 200, halo_profile_model = 'nfw'):
        
        self.mass_def = mass_def
        
        self.cluster_clmm = clmm.Modeling(massdef = mass_def, delta_mdef = delta_mdef, halo_profile_model = halo_profile_model)
        
        self.cluster_clmm.set_cosmo(self.cosmo)
        
    def _set_cluster_redshift(self,cluster_z = 0):
        
        self.cluster_z = cluster_z
        
    def _set_is_deltasigma(self, is_deltasigma = True):
        
        self.is_deltasigma = is_deltasigma
        
    def available_parameters(self):
        
        self.params = ['logm', 'concentration']
        
        self.dict = dict(logm = 13,
               concentration = 3,
               fix_logm = False, 
               fix_concentration = False,
               limit_logm = [12,16],
               limit_concentration = [0,10],
               errordef=1)
        
    def _set_free_parameters(self, free_logm = True, free_concentration = True):
        
        free = dict(logm = free_logm, concentration = free_concentration)
        
        self.free_params = []

        for param in self.params:
            
            if free[param] == True: self.free_params.append(param)
        
        r"""
        Modeling
        """ 
        
    def _set_model_to_fit(self, z_galaxy = 1):
        
        def shear_signal(r, logm, concentration):
            
            if self.is_deltasigma == False: 
                
                res =  predict_reduced_tangential_shear_z_distrib(r, logm, concentration, self.cluster_z, z_galaxy, self.cluster_clmm)
                
            else: res = predict_excess_surface_density(r, logm, concentration, self.cluster_z, self.cluster_clmm)
                
            return np.array(res)
        
        def free_c(r, p):
            
            logm, concentration = p

            return shear_signal(r, logm, concentration)
        
        def fix_c(r, p):
            
            logm = p
            
            concentration = Duffy_concentration(10.**logm, self.cluster_z, self.mass_def)
            
            return shear_signal(r, logm, concentration)
        
        if 'concentration' in self.free_params: self.model = free_c
                
        else : self.model = fix_c
                
    def _set_dict(self):
        
        dict_ = {}

        for param in self.free_params:

            dict_[param] = self.dict[param]
            dict_['limit_'+param] = self.dict['limit_'+param]
            dict_['fix_'+param] = False

        self.dic = dict_

























            
def sigmoid(r,r0,rc):
    
    r = np.array(r)

    return np.array( 1./( 1. + np.exp( - (r - r0)/rc) ) )



def Duffy_concentration(m, z_cl, massdef):
    
    r"""
    return the concentration of a cluster of mass m (Solar Mass) at given redshift z_cl (A. R. Duffy et al. (2007))
    
    """
    
    m_pivot = 2*10**12/(0.71)
    
    if massdef == 'critical':

        #A, B, C = 5.71, -0.084, -0.47
        
        A, B, C = 6.63, -0.09, -0.55 #adjusted on c_200m(M200m)

    if massdef == 'mean':
        
        A, B, C = 10.14, -0.081, -1.01
        
    return A * ( m/m_pivot )**B *( 1 + z_cl )**C
    
def log_normal_Mc_relation(c,M,z, massdef):
    
    sigma_lnc = 0.25
    
    a = np.sqrt(2*np.pi) * sigma_lnc
    
    b = (np.log(c) - np.log(Duffy_concentration(M,z,massdef))) ** 2.
         
    d = 2 * sigma_lnc ** 2.
         
    return np.array((1/c) * np.exp(-b / d) /a)

def  predict_reduced_tangential_shear_z_distrib(r, logm, c, cluster_z, z_gal, moo):
    
    r"""returns the predict reduced tangential shear at physical distance r from the cluster center of mass m
    for a collection of background galaxy redshift
    
    Parameters
    ----------
    r : array_like, float
        Rrojected radius form the cluster center in Mpc
    logm : float
        The quantity log10(M200m) where M200m is the 200m-mass of the galaxy cluster in M_\odot
    cluster_z : float
        Redshift of the galaxy cluster
    z_gal : list
        The list of background galaxy redshifts
    cosmo : astropy Table
    
    Returns
    -------
    gt_model : array_like, float
        The predicted reduced tangential shear (no units)
    """
    
    m = 10.**logm 
    
    moo.set_mass(m) 
    
    moo.set_concentration(c)

    gt_model = []
    
    for i, R in enumerate(r):
        
        z_list = np.array(z_gal[i])
        
        shear = moo.eval_reduced_shear(R, cluster_z, z_list)
        
        gt_model.append(np.mean(shear))
        
    return np.array(gt_model)


def predict_excess_surface_density(r, logm, c, cluster_z, moo):
    
    r"""returns the predict excess surface density
    
    Parameters
    ----------
    r : array_like, float
        Rrojected radius form the cluster center in Mpc
    logm : float
        The quantity log10(M200m) where M200m is the 200m-mass of the galaxy cluster in M_\odot
    cluster_z : float
        Redshift of the galaxy cluster
    z_gal : list
        The list of background galaxy redshifts
    cosmo : astropy Table
    
    Returns
    -------
    deltasigma : array_like, float
        The predicted excess surface density zero-order and second-order
    """
    m = 10.**logm 
    
    moo.set_mass(m) 
    
    moo.set_concentration(c)
    
    deltasigma = []
    
    for i, R in enumerate(r):
        
        surface_density_nfw = moo.eval_sigma_excess(R, cluster_z)
        
        deltasigma.append(surface_density_nfw)
        
    return deltasigma

def predict_excess_surface_density_concentration_scatter(r, logm, cluster_z, z_gal, moo):
    
    m = 10.**logm 
    
    moo.set_mass(m) 
    
    def integrand(c,R,M,z, massdef):
        
        moo.set_concentration(c)
        
        return moo.eval_sigma_excess(R, z) * log_normal_Mc_relation(c,M,z, massdef)
    
    for i, R in enumerate(r):
        
        surface_density_nfw = quad(integrand, 0, 50, args = (R,m,cluster_z,moo.massdef))[0]
        
        deltasigma.append(surface_density_nfw)
        
    return deltasigma

"""
def predict_convergence_z_distrib(r, logm, c, cluster_z, z_gal, moo):
    
    m = 10.**logm 
    
    c = Duffy_concentration(m, cluster_z, moo)
    
    moo.set_mass(m) 
    
    moo.set_concentration(c)
    
    kappa = []
    
    for i, R in enumerate(r):
        
        z_list = np.array(z_gal[i])
        
        kp = moo.eval_convergence(R, cluster_z, z_list)
        
        kappa.append(np.mean(kp))
        
    return np.array(kappa)

def predict_shear_z_distrib(r, logm, c, cluster_z, z_gal, moo):
    
    m = 10.**logm 
    
    c = Duffy_concentration(m, cluster_z, moo)
    
    moo.set_mass(m) 
    
    moo.set_concentration(c)
    
    signal = []
    
    for i, R in enumerate(r):
        
        z_list = np.array(z_gal[i])
        
        shear = moo.eval_shear(R, cluster_z, z_list)
        
        convergence = moo.eval_convergence (R, cluster_z, z_list)
        
        signal.append(np.mean(shear/(1 - convergence)))
        
    return np.array(signal)
    
"""

def critical_surface_density(z_l, z_s, cosmo):

    d_l = cosmo.eval_da_z1z2(0, z_l)
    
    G = const.G.to(u.Mpc**3 / (u.Msun * u.year**2))

    c = const.c.to(u.Mpc / u.year)

    second_term = (c**2/(np.pi*4*G)).value
    
    sigma_c = []
    
    for z in z_s :
    
        d_s = cosmo.eval_da_z1z2(0, z)

        d_ls = cosmo.eval_da_z1z2(z_l, z)

        first_term = d_s/(d_ls * d_s)
        
        sigma_c.append(first_term * second_term)

    return np.array(sigma_c)