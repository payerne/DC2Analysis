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
        self.LS_list = []
        self.cosmo = cosmo
        
        self.radial_axis = [[] for i in range(n_bins)]
        self.signal = [[] for i in range(n_bins)]
        self.weight = [[] for i in range(n_bins)]
        self.z_galaxy = [[] for i in range(n_bins)]
        
        self.r_low = r_low
        self.r_up = r_up
        self.n_bins = n_bins
        
        self.n_stacked_cluster = 0
        self.z_average = 0
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
            
            self.radial_axis[i].extend([R])
            
            galist = profile['gal_id'][i]
            
            sigma_c = cl.galcat['sigma_c'][galist]
            
            critical_density_2 = (sigma_c)**(-2)
            
            delta_sigma = cl.galcat['et'][galist]
            
            if self.is_deltasigma == True : signal = delta_sigma 
            
            else : signal = delta_sigma/sigma_c
                
            self.signal[i].extend(signal)
            
            if self.is_deltasigma == False : self.weight[i].extend([1 for i in range(len(galist))])
            
            else : self.weight[i].extend(critical_density_2)
            
            self.z_galaxy[i].extend(cl.galcat['z'][galist])
            
            
    def _estimate_individual_lensing_signal(self, cl, profile):

        gt_individual = []
        
        for i, R in enumerate(profile['radius']):
            
            galist = profile['gal_id'][i]
            
            sigma_c = cl.galcat['sigma_c'][galist]
            
            if self.is_deltasigma == False : weight = [1 for i in range(len(galist))]
            
            else: weight = (sigma_c)**(-2)
            
            signal = cl.galcat['et'][galist]
            
            if self.is_deltasigma == False : signal = cl.galcat['et'][galist]/sigma_c
            
            gt_individual.append(np.nansum(signal*weight)/np.nansum(weight))
            
        self.LS_list.append(np.array(gt_individual))
            

    def MakeStackedProfile(self):
        
        self.z_average = np.mean(self.z_cluster_list)
        
        gt_stack = []
        
        radius_stack = []
        
        for i in range(self.n_bins):
            
            gt_stack.append(np.nansum(np.array(self.signal[i])*np.array(self.weight[i]))/np.nansum(np.array(self.weight[i])))
            
            radius_stack.append(np.nanmean(self.radial_axis[i]))
            
        profile = Table()
        
        profile['gt'] = gt_stack
        
        profile['radius'] = radius_stack
            
        self.profile = profile
        
    def _add_standard_deviation(self):
        
        r"""
        Add standard deviation of the stacked profile to individual selected clusters 
        """

        self.profile['gt_err'] = np.sqrt(np.nanmean((self.LS_list - self.profile['gt'])**2, axis = 0))




        
        