import sys
import os
os.environ['CLMM_MODELING_BACKEND'] = 'nc' # here you may choose ccl or nc (NumCosmo)
sys.path.append('/pbs/throng/lsst/users/cpayerne/CLMM/examples/support')
try: import clmm
except:
    import notebook_install
    notebook_install.install_clmm_pipeline(upgrade=False)
    import clmm

import numpy as np
from astropy.table import Table
import fnmatch
import pickle 

import clmm.polaraveraging as pa
import clmm.galaxycluster as gc
import clmm.modeling as modeling

import glob
from scipy import interpolate

sys.path.append('/pbs/throng/lsst/users/cpayerne/GitForThesis/DC2Analysis')

import utils as ut
import statistics_ as stat


class Shear():
    
    r"""
    A class for creating stacked background galaxy catalog and deriving shear profile using stacking estimation.
    """
    
    def __init__(self, is_deltasigma = True, cosmo = 0):
        
        self.is_deltasigma = is_deltasigma
        
        self.cosmo = cosmo
        
   
    def make_binned_profile(self, metacatalog = 1, tan = 'st_true', cross = 'sx_true', weights = 'w_ls_true', bin_edges = 1):
        
        cl = metacatalog
        
        """
        Attributes:
        ----------
        cl : GalaxyCluster catalog (clmm)
            the background galaxy cluster catalog where weights are computed, and radial distance to cluster center
        bin_edges: array
            edges of radial bins for making binned profile
        Returns:
        -------
        profile : Astropy Table
            table containing shear estimation information
        """
        
        radial_bins = [[bin_edges[i], bin_edges[i + 1]] for i in range(len(bin_edges) - 1)]
        
        """stacked signal quantities"""
                  
        radius, s_t, s_x = [], [], []
        
        """assigned value to locate individual galaxies """
        
        n_gal, gal_id, r_to_center, halo_id, wls  = [], [], [], [], [] 
        
        """individual weak lensing quantities"""
        
        e_tangential, e_cross = [], []
                  
        for radial_bin in radial_bins:
                  
            mask_down, mask_up = (cl.galcat['r'] >= radial_bin[0]),  (cl.galcat['r'] < radial_bin[1])
            
            mask = mask_down * mask_up
                
            r = cl.galcat['r'][mask]
                  
            et = cl.galcat[tan][mask]
                  
            ex = cl.galcat[cross][mask]
            
            w_ls = cl.galcat[weights][mask]
                  
            radius.append(np.sum(r * w_ls)/np.sum(w_ls))
                  
            s_t.append(np.sum(et * w_ls)/np.sum(w_ls))
                  
            s_x.append(np.sum(ex * w_ls)/np.sum(w_ls))
               
            n_gal.append(len(r))
            
            gal_id.append(cl.galcat['dc2_galaxy_id'][mask])
            
            halo_id.append(cl.galcat['halo_id'][mask])
            
            wls.append(w_ls)
            
            r_to_center.append(cl.galcat['r'][mask])
            
            e_tangential.append(et)
            
            e_cross.append(ex)
            
        profile = Table()
        
        profile['gt'] = np.array(s_t)
        
        profile['gx'] = np.array(s_x)
                  
        profile['radius'] = np.array(radius)
        
        profile['n_gal'] = np.array(n_gal)
        
        profile['gal_id'] = np.array(gal_id)
        
        profile['halo_id'] = np.array(halo_id)
        
        profile['r'] = np.array(r_to_center)
        
        profile['w_ls'] = np.array(wls)
        
        profile['et'] = np.array(e_tangential)
        
        profile['ex'] = np.array(e_cross)
        
        self.profile = profile
                       
        return profile
    
    def sample_covariance(self, metacatalog = 1, bin_edges = 1):
        
        """
        Methods:
        -------
            compute the sample covariance matrix from individual shear measurements 
        Attributes:
        ----------
        bin_edges : array
            the edges of the radial bins [Mpc]
        Returns:
        -------
        self.cov_t_sample, self.cov_x_sample : array, array
            the covariance matrices respectively for tangential and cross stacked shear
        """
        
        n_bins = len(self.profile['radius'])

        Stat_t, Stat_x = stat.Statistics(n_bins), stat.Statistics(n_bins)

        for cl_individual in metacatalog.list_cl:

            self.add_weights(cl_individual)
            
            ut._add_distance_to_center(cl_individual, self.cosmo)

            profile_ = self.make_binned_profile(cl = cl_individual, bin_edges = bin_edges)

            Stat_t._add_realization(profile_['gt']), Stat_x._add_realization(profile_['gx'])

        Stat_t.estimate_covariance(), Stat_x.estimate_covariance()

        self.cov_t_sample = Stat_t.covariance_matrix * 1/(self.cl.n_stacked_catalogs - 1)

        self.cov_x_sample = Stat_x.covariance_matrix * 1/(self.cl.n_stacked_catalogs - 1)

        return self.cov_t_sample, self.cov_x_sample
    
    def bootstrap_resampling(self, binned_profile = 1, metacatalog = 1, n_boot = 1):
        
        """
        Method:
        ------
        Calculates the bootstrap covariance matrix from true shear measurements
        Attributes:
        ----------
        binned_profile : Astropy Table
            Table containing meta data and binned profile
        catalog : GalaxyCluster catalog
            meta data catalog
        n_boot : int
            the number of bootstrap resampling
        Returns:
        -------
        cov_t_boot, cov_x_boot : array, array
            the covariance matrices respectively for tangential and cross shear
        """
        
        profile = binned_profile

        Stat_t = stat.Statistics(len(profile['radius']))

        Stat_x = stat.Statistics(len(profile['radius']))

        indexes = np.arange(metacatalog.n_stacked_catalogs)

        for n in range(n_boot):

            choice_halo_id = np.array(np.random.choice(indexes, metacatalog.n_stacked_catalogs))
   
            unique_halo_id, n_repeated = np.unique(choice_halo_id, return_counts = True)

            signal_t, signal_x = [], []

            index = np.argsort(unique_halo_id)

            unique_id = unique_halo_id[index]

            repetition = n_repeated[index]

            f = interpolate.interp1d(unique_id, repetition)

            for i, r in enumerate(profile['radius']):

                mask = np.isin(profile['halo_id'][i], unique_halo_id)

                halo_id = np.array(profile['halo_id'][i][mask])

                et, ex = profile['et'][i][mask], profile['ex'][i][mask]

                r = f(halo_id)

                wls = profile['w_ls'][i][mask] * r

                signal_t.append(np.sum(et*wls)/np.sum(wls))

                signal_x.append(np.sum(ex*wls)/np.sum(wls))

            Stat_t._add_realization(signal_t), Stat_x._add_realization(signal_x)

        Stat_t.estimate_covariance(), Stat_x.estimate_covariance()

        cov_t_boot = Stat_t.covariance_matrix * 1/(n_boot - 1)

        cov_x_boot = Stat_x.covariance_matrix * 1/(n_boot - 1)

        return cov_t_boot, cov_x_boot
    
    def bootstrap_resampling_with_numpy(self, binned_profile = 1, metacatalog = 1, n_boot = 1):
        
        """
        Method:
        ------
        Calculates the bootstrap covariance matrix from true shear measurements
        Attributes:
        ----------
        binned_profile : Astropy Table
            Table containing meta data and binned profile
        catalog : GalaxyCluster catalog
            meta data catalog
        n_boot : int
            the number of bootstrap resampling
        Returns:
        -------
        cov_t_boot, cov_x_boot : array, array
            the covariance matrices respectively for tangential and cross shear
        """
        
        profile = binned_profile
        
        names = []
        
        for i, r in enumerate(profile):
            
            names.append('x' + str(i))
        
        t,x = Table(names = names), Table(names = names)

        indexes = np.arange(metacatalog.n_stacked_catalogs)

        for n in range(n_boot):

            choice_halo_id = np.array(np.random.choice(indexes, metacatalog.n_stacked_catalogs))
   
            unique_halo_id, n_repeated = np.unique(choice_halo_id, return_counts = True)

            signal_t, signal_x = [], []

            index = np.argsort(unique_halo_id)

            unique_id, repetition = unique_halo_id[index], n_repeated[index]

            f = interpolate.interp1d(unique_id, repetition)

            for i, r in enumerate(profile['radius']):

                mask = np.isin(profile['halo_id'][i], unique_halo_id)

                halo_id = np.array(profile['halo_id'][i][mask])

                et, ex = profile['et'][i][mask], profile['ex'][i][mask]

                r = f(halo_id)

                wls = profile['w_ls'][i][mask] * r

                signal_t.append(np.sum(et*wls)/np.sum(wls))

                signal_x.append(np.sum(ex*wls)/np.sum(wls))

            t.add_row(signal_t), x.add_row(signal_x)
            
        listet, listex = [], []
        
        for n in names:
            
            listet.append(t[n]), listex.append(x[n])
            
        Xt, Xx = np.stack((listet), axis=0), np.stack((listex), axis=0)
        
        covt, covx = np.cov(Xt), np.cov(Xx)
        
        return covt, covx
            
    
    
    def jacknife_resampling(self, binned_profile = 1, catalog = 1, n_jk = 1):
        
        """
        Method:
        ------
        Calculates the jacknife covariance matrix from true shear measurements
        
        Attributes:
        ----------
        binned_profile : Astropy Table
            Table containing meta data and binned profile
        catalog : GalaxyCluster catalog
            meta data catalog
        n_jk : int
            the number of jacknife resampling
            
        Returns:
        -------
        cov_t_jk, cov_x_jk : array, array
            the covariance matrices respectively for tangential and cross shear
        """
    
        profile = binned_profile
        
        cl = catalog
        
        n_bins = len(profile['radius'])

        Stat_t, Stat_x = stat.Statistics(n_bins), stat.Statistics(n_bins)

        indexes = np.arange(cl.n_stacked_catalogs)

        indexes_cut = np.array_split(indexes, n_jk)

        for i in range(n_jk):

            unique_halo_id = indexes_cut[i]

            signal_t, signal_x = [], []

            for i, R in enumerate(profile['radius']):

                mask_is_in = np.isin(profile['halo_id'][i], unique_halo_id)

                mask = np.invert(mask_is_in)

                halo_id = np.array(profile['halo_id'][i][mask])

                et, ex = profile['et'][i][mask], profile['ex'][i][mask]

                wls = profile['w_ls'][i][mask] 

                signal_t.append(np.sum(et*wls)/np.sum(wls))

                signal_x.append(np.sum(ex*wls)/np.sum(wls))

            Stat_t._add_realization(signal_t), Stat_x._add_realization(signal_x)

        Stat_t.estimate_covariance(), Stat_x.estimate_covariance()

        coeff = 1

        cov_t_jk = Stat_t.covariance_matrix  * ((n_jk - 1)/(n_jk)) * coeff

        cov_x_jk = Stat_x.covariance_matrix  * ((n_jk - 1)/(n_jk)) * coeff

        return cov_t_jk, cov_x_jk







                  
            
                  
        



                