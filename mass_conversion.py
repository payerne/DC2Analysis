import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/GitForThesis/DC2Analysis/modeling')
import Einasto_profile as ein
import NFW_profile as nfw
import Hernquist_profile as hernquist
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

def M200m_to_M200c_hernquist(M200m, c200m, z, cosmo_astropy):
    
    cl_200m = hernquist.Modeling(M200m, c200m, z, 'mean', cosmo_astropy)
    
    M200m, r200m = cl_200m.M200, cl_200m.r200
    
    def f(p):
        
        M200c, c200c = p[0], p[1]
        
        cl_200c = hernquist.Modeling(M200c, c200c, z, 'critical', cosmo_astropy)
        
        r200c = cl_200c.r200
        
        """first term"""
        
        first_term = M200c - cl_200m.M(r200c)
        
        """second term"""
        
        second_term = M200m - cl_200c.M(r200m)
        
        return first_term, second_term
    
    x0 = [M200m, c200m]
    
    M200c, c200c = fsolve(func = f, x0 = x0)
    
    return M200c, c200c