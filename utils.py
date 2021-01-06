import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/GitForThesis/DC2Analysis/modeling')
import Einasto_profile as ein
import NFW_profile as nfw
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

def e1_e2(chi1, chi2):
    
    """
    Attributes:
    ----------
    chi1, chi2 : float, float
        the two components of ellipticity (chi-definition)
        
    Returns:
    -------
    e1, e2 : float, float
       the two components of ellipticity (epsilon-definition) 
    """
    
    chi = chi1 + 1j * chi2
    
    phase = np.angle(chi)
    
    norm_chi = np.sqrt(chi1**2 + chi2**2)
    
    norm_epsilon = norm_chi / (1 + (1 - norm_chi**2) ** 0.5 )
    
    epsilon = norm_epsilon * np.exp(1j * phase)
    
    return epsilon.real, epsilon.imag

def variance_epsilon(chi1, chi2, variance_chi):
    
    """
    Attributes:
    ----------
    chi1, chi2 : float, float
        the two components of ellipticity (chi-definition)
    variance_chi : float
        the uncertainty on the absolute chi-ellipticity
        
    Returns:
    -------
    variance_epsilon : float
        the uncertainty on the absolute epsilon-ellipticity
    """

    e1, e2 = e1_e2(chi1, chi2)
    
    epsilon = np.sqrt(e1**2 + e2**2)
    
    chi = np.sqrt(chi1**2 + chi2**2)
    
    depsilon_dchi = epsilon * ( 1./chi + epsilon /( ( 1. - chi ** 1. ) ** 0.5 ) )
    
    return ( depsilon_dchi ** 2 ) * variance_chi

def _add_shapenoise(cl_stack):
    
    
    """
    Attributes:
    ----------
    cl_stack : GalaxyCluster object (clmm)
    
    Methods:
    -------
    compute lensed ellipticity e1, e2 
    with true ellipticity e1_true, e2_true and shear quantitites kappa, shear
    
    """
    
    es1, es2 = cl_stack.galcat['e1_true'], cl_stack.galcat['e2_true']
    
    gamma1, gamma2 = cl_stack.galcat['shear1'], cl_stack.galcat['shear2']

    kappa = cl_stack.galcat['kappa']
                
    e1_m = compute_lensed_ellipticity(es1, es2, gamma1, gamma2, kappa)[0]
    
    e2_m = compute_lensed_ellipticity(es1, es2, gamma1, gamma2, kappa)[1]
                
    cl_stack.galcat['e1'] = e1_m
    
    cl_stack.galcat['e2'] = e2_m
    
    return cl_stack

def _is_complete(cl, r_limit, cosmo):
    
    r"""
    
    This functions tests completness of cl for weak lensing analysis
    
    Attributes:
    ----------
    
    cl : GalaxyCluster object from CLMM
        galaxy cluster metadata containing ra, dec for each background galaxy
    r_limit : float
        upper limit for radial distance from the cluster center
        
    Returns:
    -------
    
    True, False : Boolean
        True if the cl catalog is complete (the number of empty cells <= n_empty_cells_limit)
        False if the cl catalog is not complete (the number of empty cells >= n_empty_cells_limit)
    
    """
    n_empty_cells_limit = 5
    
    rmax = r_limit
    
    rmax, dist = r_limit, cosmo.eval_da(cl.z)
    
    d_ra, d_dec = (cl.galcat['ra'] - cl.ra), (cl.galcat['dec'] - cl.dec)
    
    phi = np.arctan2(d_dec,d_ra)
    
    angle = np.sqrt(d_ra ** 2 + d_dec ** 2) * np.pi/180 # in radians
    
    x_center = dist * angle * np.cos(phi) #in Mpc
    
    y_center = dist * angle * np.sin(phi) #in Mpc
    
    n_x = 5
    
    n, width = n_x**2, 2*rmax
    
    path = width/n_x
    
    X0 = path * np.array([i for i in range(n_x)]) - width/2

    Y0 = path * np.array([i for i in range(n_x)]) - width/2
    
    cut = np.zeros((int(n_x),int(n_x)))
    
    for i, x0 in enumerate(X0):

        for j, y0 in enumerate(Y0):
            
            mask_x = (x_center > x0)  * (x_center <= x0 + path)
            
            mask_y = (y_center > y0)  * (y_center <=  y0 + path)
        
            mask = mask_x * mask_y
            
            cut[i][j] = len(x_center[mask])
            
    is_empty = (cut.flatten() <= n_empty_cells_limit)
            
    return len(cut.flatten()[is_empty]) <= n_empty_cells_limit
    

def _add_weights(cl, is_deltasigma = True):
    
        sigma_SN = 0.3
        
        r"""
        Attributes:
        ----------
        cl : GalaxyCluster catalog
            object that contains the cluster metadata
        Returns:
        -------
        Assign each galaxies in cl object with weight w_ls for stacking analysis
        """
        
        n_gal = len(cl.galcat['id'])
        
        try:  
            
            mask_redshift = np.array([pzbins_ > cl.z for i, pzbins_ in enumerate(cl.galcat['pzbins'])])

            mask_last_item = np.array([ (pzbins_ != pzbins_[-1]) for i, pzbins_ in enumerate(cl.galcat['pzbins'])])

            mask = mask_redshift * mask_last_item

            pzbins = np.array([pzbins_[mask[i]] for i, pzbins_ in enumerate(cl.galcat['pzbins'])])

            pzbins_le = np.array([pzbins_[mask_redshift[i]] for i, pzbins_ in enumerate(cl.galcat['pzbins'])])

            dz = np.array([np.array(pzbins_[1:len(pzbins_):1] - pzbins_[0:len(pzbins_) - 1:1])  for i, pzbins_ in enumerate(pzbins_le)]) 

            pdf = np.array([pzpdf_[mask[i]] for i, pzpdf_ in enumerate(cl.galcat['pzpdf'])])

            cl.galcat['sigma_c_pdf'] = modeling.critical_surface_density(cl.z, pzbins ,self.cosmo)

            norm_pdf = np.array([np.sum(pdf_*dz[i])  for i, pdf_ in enumerate(pdf)])

            critical_density_2 = np.array([np.sum(sigma_c**(-2.)*pdf[i]*dz[i]/norm_pdf[i]) \
                                           for i, sigma_c in enumerate(cl.galcat['sigma_c_pdf'])])
            
        except:
        
            sigma_c = cl.galcat['sigma_c']
            
            critical_density_2 = 1./( sigma_c ** 2.)
            
        try: sigma_epsilon = cl.galcat['rms_ellipticity']

        except: sigma_epsilon = np.zeros(n_gal) 
        
        weight_epsilon = 1./(sigma_SN ** 2 + sigma_epsilon ** 2)

        if is_deltasigma == True : w_ls = critical_density_2 * weight_epsilon
            
        else : w_ls = weight_epsilon * 1.
        
        cl.galcat['w_ls'] = w_ls 
        

def _add_distance_to_center(cl, cosmo):
    
    """
    Attributes:
    ----------
    cl : GalaxyCluster object
    cosmo : clmm Cosmology object
    
    Methods:
    -------
    add a column with physical distance to cluster center in Mpc
    
    """
    
    dx = cosmo.eval_da(cl.z)

    d_dec = (cl.galcat['dec'] - cl.dec)

    d_ra = (cl.galcat['ra'] - cl.ra)

    theta = np.arctan2(d_dec,d_ra) 

    deg = np.sqrt(d_ra ** 2 + d_dec ** 2) * np.pi/180 # in radians

    r = dx * deg

    phi = np.arctan2(d_dec,d_ra)

    cl.galcat['r'] = np.array(r)

    cl.galcat['phi'] = np.array(phi)