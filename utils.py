import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/GitForThesis/DC2Analysis/modeling')
import Einasto_profile as ein
import NFW_profile as nfw
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import fsolve
import numpy as np
import pyccl as ccl
import astropy.units as u
from scipy.special import gamma, gammainc

import sys
import os
os.environ['CLMM_MODELING_BACKEND'] = 'ccl' # here you may choose ccl or nc (NumCosmo)
sys.path.append('/pbs/throng/lsst/users/cpayerne/CLMM/examples/support')
try: import clmm
except:
    import notebook_install
    notebook_install.install_clmm_pipeline(upgrade=False)
    import clmm
    
from clmm.utils import compute_lensed_ellipticity

def M200m_to_M200c_nfw(M200m, c200m, z, cosmo_astropy):
    
    r"""
    Attributes:
    ----------
    M200m : array
        the mass M200m of the cluster
    c200m : array
        the concentration c200m associated to mass M200m of the cluster
    z : float
        cluster redshift
        
    Returns:
    -------
    M200c : array
        the mass M200c of the cluster
    c200c : array
        the concentration c200c associated to mass M200m of the cluster
    """
    
    cl_200m = nfw.Modeling(M200m, c200m, z, 'mean', cosmo_astropy)
    
    M200m, r200m = cl_200m.M200, cl_200m.r200
    
    def f(p):
        
        M200c, c200c = p[0], p[1]
        
        cl_200c = nfw.Modeling(M200c, c200c, z, 'critical', cosmo_astropy)
        
        r200c = cl_200c.r200
        
        """first term"""
        
        first_term = M200c - cl_200m.M(r200c)
        
        """second term"""
        
        second_term = M200m - cl_200c.M(r200m)
        
        return first_term, second_term
    
    x0 = [M200m, c200m]
    
    M200c, c200c = fsolve(func = f, x0 = x0)
    
    return M200c, c200c



def M200m_to_M200c_einasto(M200m, c200m, alpha200m, z, cosmo_astropy):
    
    r"""
    Attributes:
    ----------
    M200m : array
        the mass M200m of the cluster
    c200m : array
        the concentration c200m associated to mass M200m of the cluster
    alpha200m : float
        the slope parameter for200m
    z : float
        cluster redshift
        
    Returns:
    -------
    M200c : array
        the mass M200c of the cluster
    c200c : array
        the concentration c200c associated to mass M200m of the cluster
    alpha200c : float
        the slope parameter for 200c
    """
    
    cl_200m = ein.Modeling(M200m, c200m, alpha200m, z, 'mean', cosmo_astropy)
    
    M200m, c200m, alpha200m = cl_200m.M200, cl_200m.concentration, cl_200m.a
    
    r200m = cl_200m.r200
    
    Mtot = cl_200m.Mtot
        
    def f(p): 
        
        logM200c, c200c, alpha200c = p[0], p[1], p[2]
        
        M200c = 10**logM200c
        
        cl_200c = ein.Modeling(M200c, c200c, alpha200c, z, 'critical', cosmo_astropy)
        
        r200c = cl_200c.r200
        
        """first_term"""
        
        first_term = M200m - cl_200c.M(r200m)
        
        """second_term"""
        
        second_term = Mtot - cl_200c.Mtot
        
        """third_term"""
        
        third_term =  M200c - cl_200m.M(r200c)
        
        return first_term, second_term, third_term
    
    x0 = [np.log10(M200m), c200m, alpha200m]
    
    logM200c, c200c, alpha200c = fsolve(f, x0)
    
    return 10**logM200c, c200c, alpha200c

def e1_e2(chi1, chi2):
    
    chi = chi1 + 1j * chi2
    
    phase = np.angle(chi)
    
    norm_chi = np.sqrt(chi1**2 + chi2**2)
    
    norm_epsilon = norm_chi / (1 + (1 - norm_chi**2) ** 0.5 )
    
    epsilon = norm_epsilon * np.exp(1j * phase)
    
    return epsilon.real, epsilon.imag

def variance_epsilon(chi1, chi2, variance_chi):

    e1, e2 = e1_e2(chi1, chi2)
    
    epsilon = np.sqrt(e1**2 + e2**2)
    
    chi = np.sqrt(chi1**2 + chi2**2)
    
    depsilon_dchi = epsilon * ( 1./chi + epsilon /( ( 1. - chi ** 1. ) ** 0.5 ) )
    
    return ( depsilon_dchi ** 2 ) * variance_chi


def add_shapenoise(cl_stack):
    
    r"""
    Use before .compte_tangential_...
    """
    
    es1, es2 = cl_stack.galcat['e1_true'], cl_stack.galcat['e2_true']
    
    gamma1, gamma2 = cl_stack.galcat['shear1'], cl_stack.galcat['shear2']

    kappa = cl_stack.galcat['kappa']
                
    e1_m = calc_lensed_ellipticity(es1, es2, gamma1, gamma2, kappa)[0]
    
    e2_m = calc_lensed_ellipticity(es1, es2, gamma1, gamma2, kappa)[1]
                
    cl_stack.galcat['e1'] = e1_m
    
    cl_stack.galcat['e2'] = e2_m
    
    return cl_stack

def _is_complete(cl, r_limit, cosmo):
    
    n_empty_cells_limit = 5
    
    r"""
    
    This functions tests completness of cl for weak lensing analysis
    
    Attributes:
    ----------
    
    cl : GalaxyCluster object from CLMM
        galaxy cluster metadata containing ra, dec for each background galaxy
    r_limit : float
        upper limit for radial distance from the cluster center
        
    Returns:
    -------
    
    True, False : Boolean
        True if the cl catalog is complete (the number of empty cells <= n_empty_cells_limit)
        False if the cl catalog is not complete (the number of empty cells >= n_empty_cells_limit)
    
    """
    
    rmax = r_limit
    
    rmax, dist = r_limit, cosmo.eval_da(cl.z)
    
    d_ra, d_dec = (cl.galcat['ra'] - cl.ra), (cl.galcat['dec'] - cl.dec)
    
    phi = np.arctan2(d_dec,d_ra)
    
    angle = np.sqrt(d_ra ** 2 + d_dec ** 2) * np.pi/180 # in radians
    
    x_center = dist * angle * np.cos(phi) #in Mpc
    
    y_center = dist * angle * np.sin(phi) #in Mpc
    
    n_x = 5
    
    n, width = n_x**2, 2*rmax
    
    path = width/n_x
    
    X0 = path * np.array([i for i in range(n_x)]) - width/2

    Y0 = path * np.array([i for i in range(n_x)]) - width/2
    
    cut = np.zeros((int(n_x),int(n_x)))
    
    for i, x0 in enumerate(X0):

        for j, y0 in enumerate(Y0):
            
            mask_x = (x_center > x0)  * (x_center <= x0 + path)
            
            mask_y = (y_center > y0)  * (y_center <=  y0 + path)
        
            mask = mask_x * mask_y
            
            cut[i][j] = len(x_center[mask])
            
    is_empty = (cut.flatten() <= n_empty_cells_limit)
            
    return len(cut.flatten()[is_empty]) <= n_empty_cells_limit
    
def Mfof(cl_, alpha):
        
        mp = 1.6*10**9 #Msun
        
        rho_mean = cl_.rho_critical 
        
        l_limit = alpha * ( 2. * mp / ( (4*np.pi/3.) * rho_mean ) ) ** (1./3.)
        
        def f(r):
            
            mean = (3 * mp / (4 * np.pi * cl_.density(r)) )**(1./3.) * gamma(4/3)
            
            #mean = ( 2. * mp / ( (4*np.pi/3.) * cl_.density(r) ) ) ** (1./3.)
            
            return l_limit - mean
        
        r_fof = fsolve(func = f, x0 = 0.3)
        
        M_fof = cl_.M(r_fof)
        
        return M_fof   
    
    
        
        
        
        
        