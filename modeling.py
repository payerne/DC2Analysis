import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/CLMM/examples/support')
try: import clmm
except:
    import notebook_install
    notebook_install.install_clmm_pipeline(upgrade=False)
    import clmm
import numpy as np
from astropy.table import Table

import clmm.modeling as modeling

def Duffy_concentration(m, z_cl, moo):
    
    r"""
    return the concentration of a cluster of mass m (Solar Mass) at given redshift z_cl (A. R. Duffy et al. (2007))
    
    """
    
    m_pivot = 2*10**12/(moo.cosmo['h'])
    
    if moo.massdef == 'critical':

        A, B, C = 5.71, -0.084, -0.47

    if moo.massdef == 'mean':
        
        A, B, C = 10.14, -0.081, -1.01
        
    return A * ( m/m_pivot )**B *( 1 + z_cl )**C

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


def predict_excess_surface_density(r, logm, c, cluster_z, z_gal, order, moo):
    
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
        
        s = moo.eval_shear(R, cluster_z, z_list)
        
        signal.append(np.mean(s))
        
    return np.array(signal)