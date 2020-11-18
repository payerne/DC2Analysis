from NFW_profile import Modeling
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import fsolve
import numpy as np
import pyccl as ccl

def M200m_to_M200c(M200m, c200m, z):
    
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
    
    cosmo_astropy = FlatLambdaCDM(H0=71.0, Om0=0.265, Ob0 = 0.0448)
    
    cl_200m = Modeling(M200m, c200m, z, 'mean', cosmo_astropy)
    
    M200m = cl_200m.M200
        
    r200m = cl_200m.r200()
    
    def f1(r200c, z):
        
        M200c = (4*np.pi / 3) * 200 * cl_200m.rho_c(z) * r200c ** 3
        
        rs = cl_200m.rs(r200m)
        
        x = r200c/rs
        
        a = cl_200m.A(c200m) * 4 * np.pi * rs ** 3 * cl_200m.delta_c(x)
        
        res = M200c - a
        
        return np.array(res)
    
    r200c_test = fsolve(func = f1, x0 = 0.5*r200m, args=(z))[0]
    
    M200c_test = (4*np.pi / 3) * 200 * cl_200m.rho_c(z) * r200c_test ** 3
    
    def f2(c):
        
        x = r200m/r200c_test
        
        cl_200c = Modeling(M200c_test, c, z, 'critical', cosmo_astropy)
        
        b = cl_200c.A(c) * 4 * np.pi * (r200c_test/c) ** 3 * cl_200c.delta_c(c * x)
        
        return M200m - b
    
    c200c_test = fsolve(func = f2, x0 = cl_200m.concentration)[0]
    
    return M200c_test, c200c_test