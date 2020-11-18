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
        self.LS_t_list = []
        self.LS_x_list = []
        
        self.cosmo = cosmo
        
        self.radial_axis = [[] for i in range(n_bins)]
        self.signal_t = [[] for i in range(n_bins)]
        self.signal_x = [[] for i in range(n_bins)]
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
                
                self.signal_t[i].append(math.nan)
                
                self.signal_x[i].append(math.nan)
                
                self.weight[i].append(math.nan)
                
                self.radial_axis[i].append(math.nan)
                
                continue
                
            else:
            
                sigma_c = cl.galcat['sigma_c'][galist]

                critical_density_2 = 1/(sigma_c**2.)

                et = cl.galcat['et'][galist]
                ex = cl.galcat['ex'][galist]

                if self.is_deltasigma == True : signalt, signalx = et, ex

                else : signalt, signalx = et/sigma_c, ex/sigma_c

                self.signal_t[i].extend(signalt)
                self.signal_x[i].extend(signalx)

                if self.is_deltasigma == False : self.weight[i].extend([1 for i in range(len(galist))])

                else : self.weight[i].extend(critical_density_2)

                self.z_galaxy[i].extend(cl.galcat['z'][galist])
                
                self.radial_axis[i].append(R)
            
            
    def _estimate_individual_lensing_signal(self, cl, profile):

        gt_individual = []
        
        gx_individual = []
        
        for i, R in enumerate(profile['radius']):
            
            galist = np.array(profile['gal_id'][i])
            
            galist.astype(int)
            
            if len(galist) == 0 :
                
                gt_individual.append(math.nan)
                
                gx_individual.append(math.nan)
                
                continue
                
            else :
            
                sigma_c = cl.galcat['sigma_c'][galist]

                if self.is_deltasigma == False : weight = [1 for i in range(len(galist))]

                else: weight = (sigma_c)**(-2)

                et, ex = cl.galcat['et'][galist], cl.galcat['ex'][galist]

                if self.is_deltasigma == False : et, ex = cl.galcat['et'][galist]/sigma_c, cl.galcat['ex'][galist]/sigma_c

                gt_individual.append(np.nansum(et*weight)/np.nansum(weight))
                gx_individual.append(np.nansum(ex*weight)/np.nansum(weight))
            
        self.LS_t_list.append(gt_individual)
        self.LS_x_list.append(gx_individual)
            

    def MakeStackedProfile(self):
        
        self.z_average = np.mean(self.z_cluster_list)
        
        gt_stack = []
        gx_stack = []
        
        for i in range(self.n_bins):
            
            gt = np.nansum(np.array(self.signal_t[i])*np.array(self.weight[i]))/np.nansum(np.array(self.weight[i]))
            gx = np.nansum(np.array(self.signal_x[i])*np.array(self.weight[i]))/np.nansum(np.array(self.weight[i]))
            
            gt_stack.append(gt)
            gx_stack.append(gx)
            
            
        profile = Table()
        
        profile['gt'] = gt_stack
        
        profile['gx'] = gx_stack
        
        profile['radius'] = np.nanmean(self.radial_axis, axis = 1)
            
        self.profile = profile
        
    def _add_standard_deviation(self):
        
        r"""
        Add standard deviation of the stacked profile to individual selected clusters 
        """

        Stat_t = Statistics(self.n_bins)
        Stat_x = Statistics(self.n_bins)
        
        for i in range(len(self.LS_t_list)):
            
            gt, gx = self.LS_t_list[i], self.LS_x_list[i]
            
            Stat_t._add_realization(np.array(gt)), Stat_x._add_realization(np.array(gx))
            
        Stat_t.covariance(), Stat_x.covariance()
        
        self.cov_t, self.cov_x = Stat_t.covariance, Stat_x.covariance
        
        self.profile['gt_err'], self.profile['gx_err'] = np.sqrt(self.cov_t.diagonal()), np.sqrt(self.cov_x.diagonal())
        
    def _reshape_data(self):
        
        r"""
        Reshape self.profile excluding 1 halo stack
        """
        
        profile_reshape = Table()
        
        mask = np.invert(np.isnan(np.array(self.profile['gt_err']).astype(float)))
        
        profile_reshape['gt'], profile_reshape['gx'] = self.profile['gt'][mask], self.profile['gx'][mask]
        
        profile_reshape['radius'] = self.profile['radius'][mask]
        
        profile_reshape['gt_err'], profile_reshape['gx_err'] = self.profile['gt_err'][mask], self.profile['gx_err'][mask]
        
        self.profile = profile_reshape