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
            
            sigma_c = cl.galcat['sigma_c'][galist]
            
            critical_density_2 = (sigma_c)**(-2)
            
            self.weights.append(critical_density_2)
            
            delta_sigma = cl.galcat['et'][galist]
            
            if self.is_deltasigma == True : signal = delta_sigma 
            
            else : signal = delta_sigma/sigma_c
                
            self.deltasigma.append(delta_sigma)
            
            self.weight_per_bin[i] += np.sum(critical_density_2)
            
            self.unweighted_signal_per_bin[i] += np.sum(critical_density_2 * signal)
            
    def _estimate_individual_lensing_signal(self, cl, profile):

        gt_individual = []
        
        for i, R in enumerate(profile['radius']):
            
            galist = profile['gal_id'][i]
            
            sigma_c = cl.galcat['sigma_c'][galist]
            
            critical_density_2 = (sigma_c)**(-2)
            
            if self.is_deltasigma == True : delta_sigma = cl.galcat['et'][galist]
                
            else : delta_sigma = cl.galcat['et'][galist]/sigma_c
            
            gt_individual.append(np.sum(delta_sigma*critical_density_2)/np.sum(critical_density_2))
            
        self.LS_list.append(np.array(gt_individual))
            

    def MakeStackedProfile(self):
        
        for i in range(self.n_bins):
            
            self.stacked_signal[i] = self.unweighted_signal_per_bin[i] / self.weight_per_bin[i]
            
            self.radial_axis[i] = self.radial_axis[i]/self.n_stacked_cluster
            
        profile = Table()
        
        profile['gt'] = self.stacked_signal
        
        profile['radius'] = self.radial_axis
            
        self.profile = profile
        

    def _add_standard_deviation(self):

        self.profile['gt_err'] = np.sqrt(np.average((self.LS_list - self.profile['gt'])**2, axis = 0))




        
        