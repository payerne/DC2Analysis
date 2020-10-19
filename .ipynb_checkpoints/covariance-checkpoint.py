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


def _add_random_ellipticity(cl_stack):

    r"""
    Parameters:
    ----------
    cl_stack : GalaxyCluster object
        .compute_tangential_and_cross_component method is not applied
        
    Returns:
    -------
    cl_stack : GalaxyCluster object
        with random ellipticity contribution

     """
    e1 = cl_stack.galcat['e1']
    e2 = cl_stack.galcat['e2']
    
    e = e1 + 1j * e2
    
    size = len(cl_stack.galcat['e1'])
    
    cos_random = np.random.random(size) * 2 - 1
    
    sin_random = np.sqrt( 1 - cos_random ** 2 )

    e_small = 0.001 * ( cos_random + 1j * sin_random )

    cl_stack.galcat['e1'] = e_random.real
    cl_stack.galcat['e2'] = e_random.imag
    
    return cl_stack
    