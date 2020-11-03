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

import modeling as model


def random_rotate_ellipticity(cl):

    r"""
    Parameters:
    ----------
    cl : GalaxyCluster object
        compute_tangential_and_cross_component method is not applied
        
    Returns:
    -------
    cl : GalaxyCluster object
        with random ellipticity rotation (vanishing the shear signal)
     """
    
    e1, e2 = cl.galcat['e1'], cl.galcat['e2']
    
    e = e1 + 1j * e2
    
    size = len(cl.galcat['e1'])
    
    angle = np.random.random(size) * 2 * np.pi
    
    cos_random = np.cos(angle)
    
    sin_random = np.sin(angle)

    rotation = ( cos_random + 1j * sin_random )
    
    e_tot = e * rotation

    cl.galcat['e1'] = e_tot.real
    
    cl.galcat['e2'] = e_tot.imag
    
    return cl


def add_random_ellipticity(cl):

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
    e1, e2 = cl.galcat['e1'], cl.galcat['e2']
    
    e = e1 + 1j * e2
    
    size = len(cl.galcat['e1'])
    
    norm_e_random_max = 1 - np.sqrt(e1**2 + e2**2)
    
    norm_e_random = np.random.random(size) * norm_e_random_max
    
    angle = np.random.random(size) * 2 * np.pi
    
    cos_random = np.cos(angle)
    
    sin_random = np.sin(angle)

    e_small = norm_e_random * ( cos_random + 1j * sin_random )
    
    e_tot = e + e_small

    cl.galcat['e1'] = e_tot.real
    
    cl.galcat['e2'] = e_tot.imag
    
    return cl

def randomize_redshift(cl):
    

    r"""
    Parameters:
    ----------
    cl : GalaxyCluster object
        .compute_tangential_and_cross_component method is not applied
        
    Returns:
    -------
    cl : GalaxyCluster object
        with randomized redshift using pzpdf of cl.

     """
    
    if 'pzsigma' not in cl.galcat.keys():
        
        print('no implemented redshift error in your catalog !')
        
        return cl
    
    else :
        
        n = len(cl.galcat['z'])
        
        z_random = []
        
        for i in range(n):
            
            pzpdf = np.array(cl.galcat['pzpdf'][i])
            
            norm = np.sum(pzpdf)
            
            probability = pzpdf/norm
            
            pzbins = np.array(cl.galcat['pzbins'][i])
    
            z_random.append(np.random.choice( pzbins , 1 , p = probability)[0])
        
        cl.galcat['z'] = np.array(z_random)
        
        return cl
    
    
def bootstrap(cl):
    
    r"""
Parameters:
----------
cl : GalaxyCluster object
    .compute_tangential_and_cross_component method is not applied

Returns:
-------
cl : GalaxyCluster object
    with bootstrap sampled catalog cl

 """
    
    size = len(cl.galcat['z'])
    
    index = [i for i in range(size)]
    
    index_random = np.random.choice(index, size)
    
    cl_bootstrap = cl
    
    for i, j in enumerate(index_random):
        
        cl_bootstrap.galcat[j] = cl.galcat[i]
    
    return cl_bootstrap