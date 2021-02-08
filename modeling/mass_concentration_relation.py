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
from scipy.integrate import quad
from astropy import units as u
from astropy import constants as const
import pyccl as ccl

def Duffy_concentration(m, z_cl, massdef):
    
    r"""
    return the concentration of a cluster of mass m (Solar Mass) at given redshift z_cl (A. R. Duffy et al. (2007))
    
    """
    
    m_pivot = 2*10**12/(0.71)
    
    if massdef == 'critical':

        #A, B, C = 5.71, -0.084, -0.47
        
        A, B, C = 6.63, -0.09, -0.55 #adjusted on c_200m(M200m)

    if massdef == 'mean':
        
        A, B, C = 10.14, -0.081, -1.01
        
    return A * ( m/m_pivot )**B *( 1 + z_cl )**C

def DK15_concentration(m, cluster_z, massdef,  cosmo_ccl):
    
    if massdef == 'critical':
    
        deff = ccl.halos.massdef.MassDef(200, 'critical', c_m_relation=None)

        conc = ccl.halos.concentration.ConcentrationDiemer15(mdef=deff)
        
        return conc._concentration(cosmo_ccl, m, 1/(1 + cluster_z))
    
    if massdef == 'mean':
        
        
    
    
    
    
    
    