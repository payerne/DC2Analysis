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


class Perform_Stacking():
    
    r"""
    A class for creating stacked background galaxy catalog and deriving shear profile using stacking estimation.
    """
    
    def __init__(self, is_deltasigma = True, cosmo = 0):
        
        self.is_deltasigma = is_deltasigma
        
        self.cosmo = cosmo
        
    def _check_available_catalogs(self,bin_def = 'M_fof', z_bin = [1,1], obs_bin = [1,1], where_source = '1', r_lim = 1):
        
        r"""
        Attributes:
        ----------
        bin_def : string
            the definition for binning the available individual background galaxy catalogs
        z_bin : list
            the bin in redshift [z_low, z_up]
        obs_bin : list
            the bin in the choosen observable 'bin_def' [obs_low, obs_up]
        where_source : string
            the directory of individual cluster catalogs
        r_lim : array
            check completeness of individual catalogs up to r_lim [Mpc]
        Returns:
        -------
        the list of corresponding files for stacking
        
        """
        
        self.z_bin = z_bin
        
        self.bin_def = 'M_fof'
        
        self.obs_bin = obs_bin
        
        self.where_source = where_source
        
        os.chdir(self.where_source)
        
        self.file_name = np.array(glob.glob('cluster_ID_*'))
        
        index_id = np.array([int(file.rsplit('.pkl',1)[0].rsplit('_').index('ID') + 1) for file in self.file_name])
        
        index_mass = np.array([int(file.rsplit('.pkl',1)[0].rsplit('_').index('mass') + 1) for file in self.file_name])
        
        index_redshift = np.array([int(file.rsplit('.pkl',1)[0].rsplit('_').index('redshift') + 1) for file in self.file_name])
        
        index_richness = np.array([int(file.rsplit('.pkl',1)[0].rsplit('_').index('richness') + 1) for file in self.file_name])
        
        id_ = np.array([int(file.rsplit('.pkl',1)[0].rsplit('_')[index_id[i]]) for i, file in enumerate(self.file_name)])
        
        mass = np.array([float(file.rsplit('.pkl',1)[0].rsplit('_')[index_mass[i]]) for i, file in enumerate(self.file_name)])
        
        redshift = np.array([float(file.rsplit('.pkl',1)[0].rsplit('_')[index_redshift[i]]) for i, file in enumerate(self.file_name)])
        
        richness = np.array([float(file.rsplit('.pkl',1)[0].rsplit('_')[index_richness[i]]) for i, file in enumerate(self.file_name)])
        
        mask_z = (redshift > self.z_bin[0]) * (redshift < self.z_bin[1])
        
        if self.bin_def == 'M_fof':
        
            mask_obs = (mass > self.obs_bin[0]) * (mass < self.obs_bin[1])
            
        elif self.bin_def == 'richness':
                
            mask_obs = (richness > self.obs_bin[0]) * (richness < self.obs_bin[1])
            
        mask_tot = mask_z * mask_obs
        
        self.file_in_bin = self.file_name[mask_tot]
        
        self.mass_in_bin = mass[mask_tot]
        
        self.z_in_bin = redshift[mask_tot]
        
        self.richness_in_bin = richness[mask_tot]
        
        self.mask_end = []
        
        for i, file in enumerate(self.file_in_bin): 

            os.chdir(self.where_source)

            cl_source = pickle.load(open(file,'rb'))
            
            r"""
            check if catalog is empty or incomplete
            """
            
            if (len(cl_source.galcat['id']) == 0) or (not ut._is_complete(cl_source, r_lim, self.cosmo)): 
                
                self.mask_end.append(False)
                
            else: self.mask_end.append(True)
                
        self.file_in_bin = self.file_in_bin[self.mask_end]
        
        if len(self.file_in_bin) < 5 : raise ValueError("Not enough catalogs for stacking")
            
        else : print(f'n = there are {len(self.file_in_bin)} available catalogs')
            
    def _check_average_inputs(self):
        
        """
        Methods:
        -------
        Calculates average redshift and average chosen observable and associated errors
        """

        self.mass, self.mass_rms = np.mean(self.mass_in_bin[self.mask_end]), np.std(self.mass_in_bin[self.mask_end])

        self.z, self.z_rms = np.mean(self.z_in_bin[self.mask_end]), np.std(self.z_in_bin[self.mask_end])

        self.rich, self.rich_rms = np.mean(self.richness_in_bin[self.mask_end]), np.std(self.richness_in_bin[self.mask_end]) 
            
    def make_GalaxyCluster_catalog(self):
        
        """
        Make GalaxyCluster catalogs with corresponding r, phi, ra, dec, e1, e2, et, ex, z_gal, sigma_c
        galaxy related quantities
        
        Returns:
        -------
        cl : GalaxyCluster
            Galaxy cluster catalog stacked along all individual clusters
        """   
        os.chdir(self.where_source)
        
        self.list_cl = []
        
        r, phi, ra, dec, e1, e2, et, ex, z_gal, sigma_c = [], [], [], [], [], [], [], [], [], []
        
        cluster_z, cluster_ra, cluster_dec = [], [], []
        
        halo_id = []
        
        for i, file in enumerate(self.file_in_bin):

            os.chdir(self.where_source)

            cl_source = pickle.load(open(file,'rb'))
            
            cl_source.compute_tangential_and_cross_components(geometry = "flat",
                                                              shape_component1 = 'shear1',
                                                              shape_component2 = 'shear2',
                                                              tan_component = 'et', cross_component = 'ex', 
                                                              is_deltasigma = True, 
                                                              cosmo = self.cosmo)
            cl_source.galcat['halo_id'] = i
        
            ut._add_distance_to_center(cl_source, self.cosmo)
            
            halo_id.extend(cl_source.galcat['halo_id'])
            
            r.extend(cl_source.galcat['r'])
            
            phi.extend(cl_source.galcat['phi'])
            
            self.list_cl.append(cl_source)
            
            e1.extend(cl_source.galcat['e1']), e2.extend(cl_source.galcat['e2'])
            
            et.extend(cl_source.galcat['et']), ex.extend(cl_source.galcat['ex'])
            
            ra.extend(cl_source.galcat['ra']), dec.extend(cl_source.galcat['dec'])
            
            sigma_c.extend(cl_source.galcat['sigma_c'])
            
            z_gal.extend(cl_source.galcat['z'])
            
            cluster_z.append(cl_source.z) 
            
            cluster_ra.append(cl_source.ra), cluster_dec.append(cl_source.ra)
        
        gal_id = np.arange(len(e1))
            
        data = clmm.GCData([np.array(gal_id),
                            np.array(ra), 
                            np.array(dec),
                            np.array(phi),
                            np.array(r), 
                            np.array(e1), 
                            np.array(e2), 
                            np.array(et), 
                            np.array(ex),
                            np.array(sigma_c),
                            np.array(z_gal),
                            np.array(halo_id)], 
                            names = ('id', 'ra', 'dec', 'phi', 'r', 'e1', 'e2', 'et', 'ex', 'sigma_c', 'z', 'halo_id'),
                            masked = True)
        
        self.cl = clmm.GalaxyCluster('Stack', np.mean(cluster_ra), np.mean(cluster_dec), np.mean(cluster_z), data)
        
        self.cl.n_stacked_catalogs = len(self.file_in_bin)
        
        self.cl.n_galaxy = len(self.cl.galcat['id'])
        
        return self.cl

    def add_weights(self, cl):
        
        r"""
        Add weights column for weak lensing analysis
        """
        
        cl.galcat['w_ls'] = 1000/(cl.galcat['sigma_c']**2)
        
    def make_binned_profile(self, cl = 1, bin_edges = 1):
        
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
                  
        radius, signal_t, signal_x = [], [], []
        
        """assigned value to locate individual galaxies """
        
        n_gal, gal_id, r_to_center, halo_id, wls  = [], [], [], [], [] 
        
        """individual weak lensing quantities"""
        
        e_tangential, e_cross = [], []
                  
        for radial_bin in radial_bins:
                  
            mask_down, mask_up = (cl.galcat['r'] >= radial_bin[0]),  (cl.galcat['r'] < radial_bin[1])
            
            mask = mask_down * mask_up
            
            print(len(mask[mask == True]))
                
            r = cl.galcat['r'][mask]
                  
            et = cl.galcat['et'][mask]
                  
            ex = cl.galcat['ex'][mask]
            
            w_ls = cl.galcat['w_ls'][mask]
                  
            radius.append(np.sum(r * w_ls)/np.sum(w_ls))
                  
            signal_t.append(np.sum(et * w_ls)/np.sum(w_ls))
                  
            signal_x.append(np.sum(ex * w_ls)/np.sum(w_ls))
               
            n_gal.append(len(r))
            
            gal_id.append(cl.galcat['id'][mask])
            
            halo_id.append(cl.galcat['halo_id'][mask])
            
            wls.append(cl.galcat['w_ls'][mask])
            
            r_to_center.append(cl.galcat['r'][mask])
            
            e_tangential.append(et)
            
            e_cross.append(ex)
            
        profile = Table()
        
        profile['gt'] = np.array(signal_t)
        
        profile['gx'] = np.array(signal_x)
                  
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
    
    def sample_covariance(self, bin_edges = 1):
        
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

        for cl_individual in self.list_cl:

            self.add_weights(cl_individual)
            
            ut._add_distance_to_center(cl_individual, self.cosmo)

            profile_ = self.make_binned_profile(cl = cl_individual, bin_edges = bin_edges)

            Stat_t._add_realization(profile_['gt']), Stat_x._add_realization(profile_['gx'])

        Stat_t.estimate_covariance(), Stat_x.estimate_covariance()

        self.cov_t_sample = Stat_t.covariance_matrix * 1/(self.cl.n_stacked_catalogs - 1)

        self.cov_x_sample = Stat_x.covariance_matrix * 1/(self.cl.n_stacked_catalogs - 1)

        return self.cov_t_sample, self.cov_x_sample
    
    def bootstrap_resampling(self, binned_profile = 1, catalog = 1, n_boot = 1):
        
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
        
        cl = catalog

        Stat_t = stat.Statistics(len(profile['radius']))

        Stat_x = stat.Statistics(len(profile['radius']))

        indexes = np.arange(cl.n_stacked_catalogs)

        for n in range(n_boot):

            choice_halo_id = np.array(np.random.choice(indexes, cl.n_stacked_catalogs))

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







                  
            
                  
        



                