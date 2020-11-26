import NFW_profile as nfw
import Einasto_profile as ein
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import fsolve
import numpy as np
import pyccl as ccl
import astropy.units as u
from scipy.special import gamma, gammainc

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
        the slpoe parameter for 200c
    """
    
    cl_200m = ein.Modeling(M200m, c200m, alpha200m, z, 'mean', cosmo_astropy)
    
    M200m = cl_200m.M200
        
    r200m = cl_200m.r200()
    
    c200m = cl_200m.concentration
    
    alpha200m = cl_200m.a
    
    up =   1.
        
    down = gammainc( 3./alpha200m ,(2./alpha200m) * c200m ** (alpha200m) )
    
    Mtot = M200m * (up / down)
        
    def f(p): 
        
        logM200c, c200c, alpha200c = p[0], p[1], p[2]
        
        M200c = 10**logM200c
        
        cl_200c = ein.Modeling(M200c, c200c, alpha200c, z, 'critical', cosmo_astropy)
        
        r200c = cl_200c.r200()
        
        """first_term"""
        
        up =    gammainc( 3./alpha200c ,(2./alpha200c) * ( c200c * r200m/r200c ) ** alpha200c )
        
        down =  gammainc( 3./alpha200c ,(2./alpha200c) * c200c ** (alpha200c) )  
        
        first_term = M200m - M200c * (up / down)
        
        """second_term"""
        
        up =   1.
        
        down =  gammainc( 3./alpha200c ,(2./alpha200c) * c200c ** (alpha200c) )
        
        second_term = Mtot - M200c * (up / down)
        
        """third_term"""
        
        up =    gammainc( 3./alpha200m ,(2./alpha200m) * ( c200m * r200c/r200m ) ** alpha200m )
        
        down =  gammainc( 3./alpha200m ,(2./alpha200m) * c200m ** (alpha200m) )
        
        third_term =  M200c - M200m * (up / down)
        
        return first_term, second_term, third_term
    
    x0 = [np.log10(M200m), c200m, alpha200m]
    
    logM200c, c200c, alpha200c = fsolve(f, x0)
    
    return 10**logM200c, c200c, alpha200c
    
        
        
        
        
        