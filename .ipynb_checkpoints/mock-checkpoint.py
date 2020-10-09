import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/CLMM/examples/support')
try: import clmm
except:
    import notebook_install
    notebook_install.install_clmm_pipeline(upgrade=False)
    import clmm
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from astropy.table import Table
import cluster_toolkit as ct

import clmm.polaraveraging as pa
import clmm.utils as utils

import modelling as model

def _add_pdf_z(cl, sigma_z_unscaled):
    
    z_true = cl.galcat['z']
    
    z_measured = sigma_z_unscaled * (1 + z_true) * np.random.randn( len(z_true) ) + z_true
    
    cl.galcat['z'] = z_measured
 
    return 0