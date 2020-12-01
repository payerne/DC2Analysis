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
import modeling

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
        
        self.n_galaxy_per_bin = np.zeros(n_bins)
        
        self.r_low = r_low
        
        self.r_up = r_up
        
        self.n_bins = n_bins
        
        self.n_stacked_cluster = 0
        
        self.z_average = 0
        
        self.is_deltasigma = None
        
        self.profile = None
        
        self.sigma_SN = 0.3
        
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
        
    def _add_signal_columns_to(self, cl):
        
        et = cl.galcat['et']

        ex = cl.galcat['ex']

        if self.is_deltasigma == True : signalt, signalx = et, ex

        else : signalt, signalx = et/sigma_c, ex/sigma_c
        
        cl.galcat['signalt'], cl.galcat['signalx'] = signalt, signalx
        
    def _add_weights_column_to(self, cl):
        
        r"""
        Assign the corresponding weight to each galaxies
        
        """
        
        n_gal = len(cl.galcat['id'])
        
        try:  
            
            store = cl.galcat['pzbins']
            
            cl.galcat['sigma_c_pdf'] = modeling.critical_surface_density(cl.z,cl.galcat['pzbins'], self.cosmo) 

            dz = np.array([ np.array([pzbins[j + 1] - pzbins[j] for j in range(len(pzbins) - 1)])  for i, pzbins in enumerate(cl.galcat['pzbins'])])

            pdf = np.array([np.array(list(pdf).pop()) for i, pdf in enumerate(cl.galcat['pzpdf'])])

            norm_pdf = np.array([np.sum(pdf_*dz[i])  for i, pdf_ in enumerate(pdf)])

            critical_density_2 = np.array([np.sum(list(sigma_c**(-2.)).pop()*pdf[i]*dz[i]/norm_pdf[i])  for i, sigma_c in enumerate(cl.galcat['sigma_c_pdf'])])
            
        except:
        
            sigma_c = cl.galcat['sigma_c']
            
            critical_density_2 = 1./( sigma_c ** 2.)
            
        try: sigma_epsilon = cl.galcat['rms_ellipticity']

        except: sigma_epsilon = np.zeros(n_gal) 
        
        weight_epsilon = 1./(self.sigma_SN ** 2 + sigma_epsilon ** 2)

        if self.is_deltasigma == True : w_ls = critical_density_2 * weight_epsilon
            
        else : w_ls = weight_epsilon * 1.
        
        cl.galcat['w_ls'] = 1.e14 * w_ls
        
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
        
        self._add_weights_column_to(cl)
        
        self._add_signal_columns_to(cl)
        
        cl.galcat['id'] = np.arange(len(cl.galcat['z']))
        
        w_ls = cl.galcat['w_ls']
        
        signalt, signalx = cl.galcat['signalt'], cl.galcat['signalx']
        
        self.n_stacked_cluster += 1
        
        for i, R in enumerate(profile['radius']):
            
            galist = np.array(profile['gal_id'][i])
            
            galist.astype(int)
            
            self.n_galaxy_per_bin[i] += int(len(galist))
            
            if len(galist) == 0:
                
                self.signal_t[i].append(math.nan)
                
                self.signal_x[i].append(math.nan)
                
                self.weight[i].append(math.nan)
                
                self.radial_axis[i].append(math.nan)
                
                continue
                
            self.signal_t[i].extend(signalt[galist])

            self.signal_x[i].extend(signalx[galist])

            self.weight[i].extend(w_ls[galist])

            self.z_galaxy[i].extend(cl.galcat['z'][galist])

            self.radial_axis[i].append(R)
        
    def _estimate_individual_lensing_signal(self, cl, profile):

        gt_individual = []
        
        gx_individual = []
        
        cl.galcat['id'] = np.arange(len(cl.galcat['z']))
        
        w_ls = cl.galcat['w_ls']
        
        signalt, signalx = cl.galcat['signalt'], cl.galcat['signalx']
        
        for i, R in enumerate(profile['radius']):
            
            galist = np.array(profile['gal_id'][i])
            
            galist.astype(int)
            
            if len(galist) == 0 :
                
                gt_individual.append(math.nan)
                
                gx_individual.append(math.nan)
                
                continue
                
            et = signalt[galist]

            ex = signalx[galist]

            weight = w_ls[galist]
           
            gt_individual.append(np.nansum(et*weight)/np.nansum(weight))
                
            gx_individual.append(np.nansum(ex*weight)/np.nansum(weight))
            
        self.LS_t_list.append(gt_individual)
        
        self.LS_x_list.append(gx_individual)
            

    def MakeStackedProfile(self):
        
        self.z_average = np.mean(self.z_cluster_list)
        
        gt_stack, gx_stack  = [], []
        
        for i in range(self.n_bins):
            
            gt = np.nansum(np.array(self.signal_t[i])*np.array(self.weight[i]))/np.nansum(np.array(self.weight[i]))
            
            gx = np.nansum(np.array(self.signal_x[i])*np.array(self.weight[i]))/np.nansum(np.array(self.weight[i]))
            
            gt_stack.append(gt), gx_stack.append(gx)
            
        self.profile = Table()
        
        self.profile['radius'] = np.nanmean(self.radial_axis, axis = 1)
        
        self.profile['gt'] = gt_stack
        
        self.profile['gx'] = gx_stack
        
        self.profile['n_gal'] = self.n_galaxy_per_bin
        
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
        
        mask = np.isnan(self.profile['gt_err'])
        
        index_nan = np.arange(self.n_bins)[mask]
        
        self.profile.remove_rows(index_nan)