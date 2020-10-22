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

def make_gt_profile(cl_stack, down, up, n_bins, is_deltasigma, cosmo):
    
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

    #profile['gt'] = [np.nan if (math.isnan(i)) else i for i in profile['gt']]
    
    #profile['gt_err'] = [np.nan if math.isnan(profile['gt'][i]) else err for i,err in enumerate(profile['gt_err'])]
    
    #profile['radius'] = [np.nan if math.isnan(profile['gt'][i]) else radius for i,radius in enumerate(profile['radius'])]

    return profile


def _stacked_signal(cl, profile):
        
        
        r"""
        Parameters:
        ---------
        cl : gc
            background galaxy catalog
        profile :
            profile
            
        Returns:
        -------
        Lensing signal for individual cluster
        
        """
        gt_individual = []
        
        for i, R in enumerate(profile['radius']):
            
            galist = profile['gal_id'][i]
            
            critical_density_2 = (cl.galcat['sigma_c'][galist])**(-2)
            
            delta_sigma = cl.galcat['et'][galist]
            
            gt_individual.append(np.sum(delta_sigma*critical_density_2)/np.sum(critical_density_2))
            
        return np.array(gt_individual)
            


class Stacking():
    
    r"""
    Object that contains the stacked galaxy cluster metadata and backgound galaxy relevant quantities for reduced tangential shear and/or excess surface density
    
    Attributes
    ---------
    z_cluster_list : array_like
        list of selected cluster redshifts
    z_galaxy_list : array_like
        list of all selected backgound galaxy redshifts
    gt_list : array_like
        list of type-chosen profile
    variance_gt_list : array_like
        list of type-chosen profile variance
        
    cosmo : astropy
        input cosmology
        
    profile : Table
        Table containing Stacked profile calculation, error and radial axis 
    r_low, r_up : float like
        lower and upper limit for the radial axis (projected physical distance to cluster center) in Mpc
    n_bins : int type
        number of bins to make binned profile
    
    """
    
    def __init__(self, r_low, r_up, n_bins, cosmo):
        
        self.z_cluster_list = []
        self.radial_axis = []
        
        self.z_galaxy_list = []
        self.weights = []
        self.deltasigma = []
        self.cosmo = cosmo
        
        self.radial_axis = np.zeros(n_bins)
        self.weight_per_bin = np.zeros(n_bins)
        self.unweighted_signal_per_bin = np.zeros(n_bins)
        self.stacked_signal = np.zeros(n_bins)
        
        self.r_low = r_low
        self.r_up = r_up
        self.n_bins = n_bins
        
        self.n_stacked_cluster = 0
        self.average_z = 0
        self.is_deltasigma = None
        self.profile = None
        
        
    def _select_type(self, is_deltasigma = True):
        
        """Indicates the type of profile to bin"""
        
        self.is_deltasigma = is_deltasigma
        
        
    def _add_cluster_redshift(self, cl):
        
        self.z_cluster_list.append(cl.z)
        
        
    def _add_background_galaxies(self, cl, profile):
        
        self.n_stacked_cluster += 1
        
        self.z_galaxy_list.extend(cl.galcat['z'])
        
        for i, R in enumerate(profile['radius']):
            
            r = profile['radius'][i]
            
            self.radial_axis[i] += r
            
            galist = profile['gal_id'][i]
            
            critical_density_2 = (cl.galcat['sigma_c'][galist])**(-2)
            
            self.weights.append(critical_density_2)
            
            delta_sigma = cl.galcat['et'][galist]
            
            self.deltasigma.append(delta_sigma)
            
            self.weight_per_bin[i] += np.sum(critical_density_2)
            
            self.unweighted_signal_per_bin[i] += np.sum(critical_density_2 * delta_sigma)
            

    def MakeStackedProfile(self):
        
        for i in range(self.n_bins):
            
            self.stacked_signal[i] = self.unweighted_signal_per_bin[i] / self.weight_per_bin[i]
            
            self.radial_axis[i] = self.radial_axis[i]/self.n_stacked_cluster
            
        profile = Table()
        
        profile['gt'] = self.stacked_signal
        
        profile['radius'] = self.radial_axis
            
        self.profile = profile
        

    def _add_standard_deviation(self):

        r"""
        Returns:
        -------
        add profile['gt_err'] column to self.profile as the weighted standard deviation (preliminary)
        """

        std = []

        for i in range(self.n_bins):

            gt_variance = np.std((self.deltasigma[i]))**2

            std.append(np.sqrt(gt_variance))

        self.profile['gt_err'] = np.array(std)




        
        