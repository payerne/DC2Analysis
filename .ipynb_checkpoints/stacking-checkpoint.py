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
import math as math

import clmm.polaraveraging as pa
import clmm.galaxycluster as gc
import clmm.modeling as modeling
from clmm import Cosmology 

sys.path.append('/pbs/throng/lsst/users/cpayerne/GitForThesis/DC2Analysis')

from statistics_ import Statistics

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
        
        r"""
        Attributes:
        ----------
        is_deltasigma : boolean
            True if Excess Surface Density, False if Reduced Tangential Shear
        Returns:
        -------
        modify self.is_deltasigma to is_deltasigma
        """
        
        self.is_deltasigma = is_deltasigma
        
        
    def _add_cluster_redshift(self, cl):
        
        r"""
        Attributes:
        ----------
        cl : GalaxyCluster catalog
            object that contains the cluster metadata
        Returns:
        -------
        add the individual cluster redshift to the list of selected cluster for stacking
        """
        
        self.z_cluster_list.append(cl.z)
        
        
    def _add_background_galaxies(self, cl, profile):
        
        r"""
        Attributes:
        ----------
        cl : GalaxyCluster catalog
            galaxy cluster metadata
        profile : Table()
            table of binned profile (CLMM)
        Returns:
        -------
        Add individual cluster background galaxy redshifts to each bin in radius
        Add individual cluster weights to stack //
        Add individual cluster lensing quantities to stack //
        Add +1 to self.n_stacked_cluster
        """
        
        self.n_stacked_cluster += 1
        
        for i, R in enumerate(profile['radius']):
            
            galist = np.array(profile['gal_id'][i])
            
            galist.astype(int)
            
            if len(galist) == 0:
                
                self.signal[i].append(math.nan)
                
                self.weight[i].append(math.nan)
                
                self.radial_axis[i].append(math.nan)
                
                continue
                
            else:
            
                sigma_c = cl.galcat['sigma_c'][galist]

                critical_density_2 = 1/(sigma_c**2.)

                delta_sigma = cl.galcat['et'][galist]

                if self.is_deltasigma == True : signal = delta_sigma 

                else : signal = delta_sigma/sigma_c

                self.signal[i].extend(signal)

                if self.is_deltasigma == False : self.weight[i].extend([1 for i in range(len(galist))])

                else : self.weight[i].extend(critical_density_2)

                self.z_galaxy[i].extend(cl.galcat['z'][galist])
                
                self.radial_axis[i].append(R)
            
            
    def _estimate_individual_lensing_signal(self, cl, profile):

        gt_individual = []
        
        for i, R in enumerate(profile['radius']):
            
            galist = np.array(profile['gal_id'][i])
            
            galist.astype(int)
            
            if len(galist) == 0 :
                
                gt_individual.append(math.nan)
                
                continue
                
            else :
            
                sigma_c = cl.galcat['sigma_c'][galist]

                if self.is_deltasigma == False : weight = [1 for i in range(len(galist))]

                else: weight = (sigma_c)**(-2)

                signal = cl.galcat['et'][galist]

                if self.is_deltasigma == False : signal = cl.galcat['et'][galist]/sigma_c

                gt_individual.append(np.nansum(signal*weight)/np.nansum(weight))
                
            
        self.LS_list.append(gt_individual)
            

    def MakeStackedProfile(self):
        
        self.z_average = np.mean(self.z_cluster_list)
        
        gt_stack = []
        
        for i in range(self.n_bins):
            
            gt = np.nansum(np.array(self.signal[i])*np.array(self.weight[i]))/np.nansum(np.array(self.weight[i]))
            
            gt_stack.append(gt)
            
            
        profile = Table()
        
        profile['gt'] = gt_stack
        
        profile['radius'] = np.nanmean(self.radial_axis, axis = 1)
            
        self.profile = profile
        
    def _add_standard_deviation(self):
        
        r"""
        Add standard deviation of the stacked profile to individual selected clusters 
        """

        Stat = Statistics(self.n_bins)
        
        for i, gt in enumerate(self.LS_list):
            
            Stat._add_realization(np.array(gt))
            
        Stat.covariance()
        
        self.cov = Stat.covariance
        
        self.profile['gt_err'] = np.sqrt(self.cov.diagonal())
        
    def _reshape_data(self):
        
        r"""
        Reshape self.profile excluding 1 halo stack
        """
        
        profile_reshape = Table()
        
        mask = np.invert(np.isnan(np.array(self.profile['gt_err']).astype(float)))
        
        profile_reshape['gt'] = self.profile['gt'][mask]
        
        profile_reshape['radius'] = self.profile['radius'][mask]
        
        profile_reshape['gt_err'] = self.profile['gt_err'][mask]
        
        self.profile = profile_reshape