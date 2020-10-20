import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/CLMM/examples/support')
try: import clmm
except:
    import notebook_install
    notebook_install.install_clmm_pipeline(upgrade=False)
    import clmm
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import numpy as np
from astropy.table import Table
import math

import clmm.polaraveraging as pa
import clmm.utils as utils

import modelling as model


def random_ellipticity(cl):

    r"""
    Parameters:
    ----------
    cl : GalaxyCluster object
        .compute_tangential_and_cross_component method is not applied
        
    Returns:
    -------
    cl : GalaxyCluster object
        with random ellipticity contribution (with local constraint norm(epsilon) < 1)

     """
    e1 = cl.galcat['e1']
    e2 = cl.galcat['e2']
    
    e = e1 + 1j * e2
    
    size = len(cl.galcat['e1'])
    
    norm_e_random_max = 1 - np.sqrt(e1**2 + e2**2)
    
    norm_e_random = np.random.random() * norm_e_random_max
    
    cos_random = np.random.random(size) * 2 - 1
    
    sin_random = np.sqrt( 1 - cos_random ** 2 )

    e_small = norm_e_random * ( cos_random + 1j * sin_random )
    
    e_tot = e + e_small

    cl.galcat['e1'] = e.real
    cl.galcat['e2'] = e.imag
    
    return cl

def cov(x,y):
    
    r"""
    Attributes:
    ----------
    
    x : array like 
        (realization of the random variable X)
    y : array like 
    (realization of the random variable X)
    
    Returns:
    -------
    
    the covariance of x and y
    
    """
    
    x, y = np.array(x), np.array(y)
    
    covariance = np.mean( (x-np.mean(x)) * (x - np.mean(y)) )
    
    return covariance

def random_redshift(cl):

    r"""
    Parameters:
    ----------
    cl : GalaxyCluster object
        .compute_tangential_and_cross_component method is not applied
        
    Returns:
    -------
    cl : GalaxyCluster object
        with random ellipticity contribution (with local constraint norm(epsilon) < 1)

     """
    z = cl.galcat['z']
    
    return cl

