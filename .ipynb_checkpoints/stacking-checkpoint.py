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

def make_gt_profile(cl_stack, down, up, n_bins, is_deltasigma, cosmo):
    
"""
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

    bin_edges = pa.make_bins(down, up , n_bins, method='evenlog10width')

    profile = cl_stack.make_binned_profile("radians", "Mpc", bins=bin_edges,cosmo=cosmo,include_empty_bins= True,gal_ids_in_bins=True)

    profile['gt'] = [np.nan if (math.isnan(i)) else i for i in profile['gt']]
    
    profile['gt_err'] = [np.nan if math.isnan(profile['gt'][i]) else err for i,err in enumerate(profile['gt_err'])]
    
    profile['radius'] = [np.nan if math.isnan(profile['gt'][i]) else radius for i,radius in enumerate(profile['radius'])]

    return profile


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
        self.z_galaxy_list = []
        self.gt_list = []
        self.variance_gt_list = []
        self.radial_axis_list = []
        self.cosmo = cosmo
        self.profile = 0
        
        self.r_low = r_low
        self.r_up = r_up
        self.n_bins = n_bins
        
        self.n_stacked_gt = 0
        self.average_z = 0
        self.is_deltasigma = None
        
        
    def SelectType(self, is_deltasigma = True):
        
        """Indicates the type of profile to bin"""
        
        self.is_deltasigma = is_deltasigma
        
    def _add_cluster_z(self, gc):
        
        self.z_cluster_list.append(gc.z)
        
    def _add_background_galaxy_z(self, gc):
        
        self.z_galaxy_list.extend(gc.galcat['z'])
        
    def AddProfile(self, profile):
        
        """add individual binned profile from individual galaxy catalog"""
        
        if self.is_deltasigma == None:
            
            raise ValueError(f"type of profile not defined yet ! available : tangential reduced shear or excess surface density")
    
        self.gt_list.append(profile['gt'])
        
        self.variance_gt_list.append(np.nan_to_num(profile['gt_err']**2, nan = np.nan, posinf = np.nan, neginf = np.nan))
        
        self.radial_axis_list.append(profile['radius'])
        
        self.n_stacked_gt += 1
        
        
    def MakeStackedProfile(self, method):
        
        """Calculates the stacked profile from individual profiles"""
        
        if self.n_stacked_gt in [0,1]:
            
            raise ValueError(f"Problem for making stack strategy : {self.n_stacked_gt} loaded galaxy catalog(s)")
    
        gt = np.array(self.gt_list)
        variance_gt = np.array(self.variance_gt_list) 
        radius = np.array(self.radial_axis_list)
        
        if method == 'error weighted':
            
            w = np.nan_to_num(1/(np.array(variance_gt)), nan = np.nan, posinf = np.nan)
            
        elif method == 'classical': 
            
            w = np.nan_to_num(1*(gt/gt), nan = np.nan, posinf = np.nan)
            
        weight = w/np.nansum(w, axis = 0)
            
        profile = Table()
            
        gt_average = np.nansum(gt*weight, axis=0)
        
        gt_average_dispersion = np.sqrt(np.nansum(weight*(gt - gt_average)**2,axis=0))
        
        radius_average = np.nansum(weight*radius, axis=0)
        
        radius_average_dispersion = np.sqrt(np.nansum(weight*(radius - radius_average)**2,axis=0))

        profile['gt'] = gt_average
        profile['gt_err'] = gt_average_dispersion
        profile['radius'] = radius_average
        profile['radius_err'] = radius_average_dispersion
        
        self.average_z = np.mean(self.z_cluster_list)
        self.profile = profile
            
            
        
class StackingWeight():
    
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
        
    def SelectType(self, is_deltasigma = True):
        
        """Indicates the type of profile to bin"""
        
        self.is_deltasigma = is_deltasigma
        
    def _add_cluster_z(self, gc):
        
        self.z_cluster_list.append(gc.z)
        
    def _add_ellipticity(self, cl, profile):
        
        self.n_stacked_cluster += 1
        
        for i, R in enumerate(profile['radius']):
            
            r = profile['radius'][i]
            self.radial_axis[i] += i
            
            galist = profile['gal_id'][i]
            critical_density_2 = (cl.galcat['sigma_c'][galist])**(-2)
            delta_sigma = cl.galcat['et'][galist]
            
            self.weight_per_bin[i] += np.sum(critical_density_2)
            self.unweighted_signal_per_bin[i] += np.sum(critical_density_2 * delta_sigma)
            
        
    def MakeStackedProfile(self):
        
        for i in range(self.n_bins):
            
            self.stacked_signal[i] = self.unweighted_signal_per_bin[i] / self.weight_per_bin[i]
            
            self.radial_axis[i] = self.radial_axis[i]/self.n_stacked_cluster
            
        profile = Table()
        profile['gt'] = self.stacked_signal
        profile['radius'] = self.radial_axis
        profile['gt_err'] = np.zeros(self.n_bins)
            
        self.profile = profile
        
        
        

        
    
        
        
        
        
        
        
        
        
        
    
        

    