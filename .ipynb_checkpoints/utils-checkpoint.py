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
import clmm.galaxycluster as gc
import clmm.modeling as modeling
from clmm import Cosmology 

def make_gt_profile(cl_stack, down, up, n_bins, cosmo):
    
    r"""
    Parameters:
    ----------
    
    cl_stack : GalaxyCluster object
        .compute_tangential_and_cross_component method from GalaxyCluster has been applied
    down, up : float
        lower and upper limit for making binned profile
    n_bins : float
        number of bins for binned profiles
    is_deltasigma : Boolean
        True is excess surface density is choosen, False is reduced tangential shear is choosen
    cosmo : Astropy table
        input cosmology
    
    Returns:
    -------
    
    profile : Table
        profile with np.nan values for empty gt, gt_err, radius bins
    
    """
    
    bin_edges = pa.make_bins( down , up , n_bins , method='evenlog10width')

    profile = cl_stack.make_binned_profile("radians", "Mpc", bins=bin_edges,cosmo=cosmo,include_empty_bins= True,gal_ids_in_bins=True)

    profile['gt'] = [np.nan if (math.isnan(i)) else i for i in profile['gt']]
    
    profile['gt_err'] = [np.nan if math.isnan(profile['gt'][i]) else err for i,err in enumerate(profile['gt_err'])]
    
    profile['radius'] = [np.nan if math.isnan(profile['gt'][i]) else radius for i,radius in enumerate(profile['radius'])]

    return profile