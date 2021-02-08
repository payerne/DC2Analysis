import numpy as np
import pyccl as ccl
from scipy.integrate import quad
from scipy.special import jv

def compute_Pk(kk, z, cosmo_ccl):
    r"""
    Atributes:
    ---------
    kk : array, float
        wave vector in Mpc^(-1)
    z : float
        cluster redshift
    cosmo_ccl : cosmology object from ccl
    Returns:
    -------
    Pk : The computed linear matter power spectrum for each kk
    """
    Pk = []
    
    for k in kk:
        
        Pk.append(ccl.linear_matter_power(cosmo_ccl, k, 1/(1+z)))
        
    return np.array(Pk)

def two_halo_term_unbaised(r, cluster_z, cosmo_ccl, kk, Pk):
    r"""
    Attributes:
    ----------
    r : array, float
        radius from cluster center in Mpc
    cluster_z: float
        cluster redshift
    cosmo_ccl : cosmology object from ccl
    kk, Pk : wavevector and corresponding linear matter power spectrum
    Return:
    ------
    
    """
    
    Da = ccl.angular_diameter_distance(cosmo_ccl, 1, 1./(1. + cluster_z))
    
    rho_m = ccl.rho_x(cosmo_ccl, 1./(1. + cluster_z), 'matter', is_comoving=False)
    
    def k_func( l ):
        
        return l / ( ( 1 + cluster_z ) * Da )

    def __integrand__( l , theta ):
        
        k = k_func( l )
        
        return l * jv( 2 , l * theta ) * np.interp( k , kk , Pk )
    
    two_h = []
    
    for i, R in enumerate(r):
        
        theta = R/Da
    
        val = quad( __integrand__  , 2*1e-4 , 20000 , args = ( theta ))[0]
        two_h.append(val * rho_m / ( 2 * np.pi  * ( 1 + cluster_z )**3 * Da**2 ))
        
    return np.array(two_h)

def halo_bais(logm = 1, mdef = 'matter', Delta = 200, cluster_z = 1, cosmo_ccl = 1):
    
    definition = ccl.halos.massdef.MassDef(Delta, mdef, c_m_relation=None)
    
    halobais = ccl.halos.hbias.HaloBiasTinker10(cosmo_ccl, mass_def=definition, mass_def_strict=True)
    
    return halobais.get_halo_bias(cosmo_ccl, 10**logm, 1/(1+cluster_z), mdef_other = definition)