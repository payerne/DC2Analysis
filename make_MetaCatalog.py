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
from astropy.table import Table, vstack
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

def load(filename, **kwargs):
    
    """Loads GalaxyCluster object to filename using Pickle"""
    
    with open(filename, 'rb') as fin:
        
        return pickle.load(fin, **kwargs)


class MetaCatalog():
    
    r"""
    A class for creating stacked background galaxy catalog and deriving shear profile using stacking estimation.
    """
    
    def __init__(self, is_deltasigma = True, cosmo = 0):
        
        self.is_deltasigma = is_deltasigma
        
        self.cosmo = cosmo
        
    def _check_available_catalogs(self,dc2_infos = 1, bin_def = 'M_fof', z_bin = [1,1], obs_bin = [1,1], where_source = '1', r_lim = 1):
        
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
        
        self.z_bin, self.obs_bin, self.bin_def  = z_bin, obs_bin, bin_def
        
        self.where_source = where_source
        
        os.chdir(self.where_source)
        
        self.file_name = np.array(glob.glob('cluster_ID_*'))
        
        index_id = np.array([int(file.rsplit('.pkl',1)[0].rsplit('_').index('ID') + 1) for file in self.file_name])
        
        id_directory = np.array([int(file.rsplit('.pkl',1)[0].rsplit('_')[index_id[i]]) for i, file in enumerate(self.file_name)])
        
        halo_id, redshift, mass, richness = dc2_infos['halo_id'], dc2_infos['z'], dc2_infos['halo_mass'], dc2_infos['richness']
    
        mask_z = (redshift > self.z_bin[0]) * (redshift < self.z_bin[1])
        
        if self.bin_def == 'M_fof':
        
            mask_obs = (mass > self.obs_bin[0]) * (mass < self.obs_bin[1])
            
        elif self.bin_def == 'richness':
                
            mask_obs = (richness > self.obs_bin[0]) * (richness < self.obs_bin[1])
            
        mask_tot = mask_z * mask_obs
        
        selected_halo_id = halo_id[mask_tot]
        
        mask_directory = np.isin(id_directory, selected_halo_id)
        
        self.file_in_bin = self.file_name[mask_directory]
        
        self.mask_end = []
        
        for i, file in enumerate(self.file_in_bin): 

            cl_source = load(file)
            
            r"""
            check if catalog is empty or incomplete
            """
            
            if (len(cl_source.galcat) == 0) or (not ut._is_complete(cl_source, r_lim, self.cosmo)): 
                
                self.mask_end.append(False)
                
            else: 
                
                self.mask_end.append(True)
                
        self.file_in_bin = np.array(self.file_in_bin)[self.mask_end]
        
        if len(self.file_in_bin) < 5 : raise ValueError("Not enough catalogs for stacking")
            
        else : print(f'n = there are {len(self.file_in_bin)} available catalogs')
            
            
    def _check_available_catalogs_test(self,dc2_infos = 1, bin_def = 'M_fof', z_bin = [1,1], obs_bin = [1,1], where_source = '1', r_lim = 1):

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

        self.z_bin, self.obs_bin, self.bin_def  = z_bin, obs_bin, bin_def

        self.where_source = where_source

        halo_id, redshift, mass, richness = dc2_infos['halo_id'], dc2_infos['z'], dc2_infos['halo_mass'], dc2_infos['richness']

        mask_z = (redshift > self.z_bin[0]) * (redshift < self.z_bin[1])

        if self.bin_def == 'M_fof':

            mask_obs = (mass > self.obs_bin[0]) * (mass < self.obs_bin[1])

        elif self.bin_def == 'richness':

            mask_obs = (richness > self.obs_bin[0]) * (richness < self.obs_bin[1])

        mask_tot = mask_z * mask_obs

        self.selected_halo_id = halo_id[mask_tot]
        
        self.file_in_bin = [where_source + '/cluster_ID_' + str(h_id) + '.pkl' for h_id in self.selected_halo_id]
                
        mask_end = []
        
        for i, file in enumerate(self.file_in_bin): 
            
            cl_source = load(file)

            r"""
            check if catalog is empty or incomplete
            """

            if (len(cl_source.galcat) == 0) or (not ut._is_complete(cl_source, r_lim, self.cosmo)): 

                mask_end.append(False)

            else: 

                mask_end.append(True)

        self.file_in_bin = np.array(self.file_in_bin)[mask_end]

        if len(self.file_in_bin) < 5 : raise ValueError("Not enough catalogs for stacking")

        else : print(f'n = there are {len(self.file_in_bin)} available catalogs')
            
    def _check_average_inputs(self, dc2_infos = 1):
        
        """
        Methods:
        -------
        Calculates average redshift and average chosen observable and associated errors
        """
        
        mask = np.isin(dc2_infos['halo_id'], self.selected_halo_id)
        
        self.mass_in_bin = dc2_infos['halo_mass'][mask]
        
        self.z_in_bin = dc2_infos['z'][mask]
        
        self.richness_in_bin = dc2_infos['richness'][mask]
        
        self.mass, self.mass_rms = np.mean(self.mass_in_bin), np.std(self.mass_in_bin)

        self.z, self.z_rms = np.mean(self.z_in_bin), np.std(self.z_in_bin)

        self.rich, self.rich_rms = np.mean(self.richness_in_bin), np.std(self.richness_in_bin) 
            
    def make_GalaxyCluster_catalog(self, shape_component1 = 'shear1', shape_component2 = 'shear2', tan_component = 'et', cross_component = 'ex'):
        
        """
        Make GalaxyCluster catalogs with corresponding r, phi, ra, dec, e1, e2, et, ex, z_gal, sigma_c
        galaxy related quantities
        
        Returns:
        -------
        cl : GalaxyCluster
            Galaxy cluster catalog stacked along all individual clusters
        """   
        
        r, et, ex, z_true, dc2_galaxy_id, id_, halo_id, sigma_c = [], [], [], [], [], [], [], []
        
        sigma_c_1_v1, odds_v1 = [], []
        
        sigma_c_1_flexzboost_v1, odds_flexzboost_v1 = [], []
        
        cluster_z, cluster_ra, cluster_dec = [], [], []
        
        for i, file in enumerate(self.file_in_bin):
            
            
            cl_source = load(file)
            
            cl_source.compute_tangential_and_cross_components(geometry = "flat",
                                                              shape_component1 = shape_component1,
                                                              shape_component2 = shape_component2,
                                                              tan_component = tan_component, cross_component = cross_component, 
                                                              is_deltasigma = True, 
                                                              cosmo = self.cosmo)
            
            cl_source.galcat['id'] = np.arange(len(cl_source.galcat))
            
            cl_source.galcat[tan_component] = cl_source.galcat[tan_component]/cl_source.galcat['sigma_c']
            
            cl_source.galcat[cross_component] = cl_source.galcat[cross_component]/cl_source.galcat['sigma_c']
            
            cl_source.galcat['halo_index'] = i
            
            ut._add_distance_to_center(cl_source, self.cosmo)
            
            r.extend(cl_source.galcat['r'])
            
            et.extend(cl_source.galcat[tan_component]), ex.extend(cl_source.galcat[cross_component])
            
            sigma_c.extend(cl_source.galcat['sigma_c'])
            
            sigma_c_1_v1.extend(cl_source.galcat['<sigma_c-1>_v1']), sigma_c_1_flexzboost_v1.extend(cl_source.galcat['<sigma_c-1>_flexzboost_v1'])
            
            odds_v1.extend(cl_source.galcat['z_odds_photoz_v1']), odds_flexzboost_v1.extend(cl_source.galcat['z_odds_photoz_flexzboost_v1'])
            
            z_true.extend(cl_source.galcat['z'])
            
            dc2_galaxy_id.extend(cl_source.galcat['dc2_galaxy_id']), halo_id.extend(cl_source.galcat['halo_index'])
            
        t = Table()
        t['r'] = np.array(r)
        t['et'] = np.array(et)
        t['ex'] = np.array(ex)
        t['z'] = np.array(z_true)
        t['dc2_galaxy_id'] = np.array(dc2_galaxy_id)
        t['halo_id'] = np.array(halo_id)
        t['sigma_c'] = np.array(sigma_c)
        t['<sigma_c-1>_v1'] = np.array(sigma_c_1_v1)
        t['z_odds_photoz_v1'] = np.array(odds_v1)
        t['<sigma_c-1>_flexzboost_v1'] = np.array(sigma_c_1_flexzboost_v1)
        t['z_odds_photoz_flexzboost_v1'] = np.array(odds_flexzboost_v1)
        
        data = clmm.GCData(t)
        
        self.cl = clmm.GalaxyCluster('Stack', 0, 0, np.mean(cluster_z), data)
        
        self.cl.n_stacked_catalogs = len(self.file_in_bin)
        
        self.cl.n_galaxy = len(self.cl.galcat['halo_id'])
    
    def compute_signal(self, photoz = 1, catalog = 1):
        
    
        if photoz == False:
    
            catalog.galcat['st_true'] = catalog.galcat['sigma_c']*catalog.galcat['et']
            catalog.galcat['sx_true'] = catalog.galcat['sigma_c']*catalog.galcat['ex']
        
        elif photoz == 'BPZ' : 
                        
            catalog.galcat['st_v1'] = catalog.galcat['<sigma_c-1>_v1']**(-1.)*catalog.galcat['et']
            catalog.galcat['sx_v1'] = catalog.galcat['<sigma_c-1>_v1']**(-1.)*catalog.galcat['ex']
            
        elif photoz == 'fleXZboost':
            
            catalog.galcat['st_flexzboost_v1'] = catalog.galcat['<sigma_c-1>_flexzboost_v1']**(-1.)*catalog.galcat['et']
            catalog.galcat['sx_flexzboost_v1'] = catalog.galcat['<sigma_c-1>_flexzboost_v1']**(-1.)*catalog.galcat['ex']
            
        return catalog
    
    def compute_weights(self, photoz = 1, catalog = 1):
        
        if photoz == False:
        
            catalog.galcat['w_ls_true'] = catalog.galcat['sigma_c']**(-2.)
        
        elif photoz == 'BPZ': 
            
            catalog.galcat['w_ls_v1'] = catalog.galcat['<sigma_c-1>_v1']**2.
            
        elif photoz == 'fleXZboost':
            
            catalog.galcat['w_ls_flexzboost_v1'] = catalog.galcat['<sigma_c-1>_flexzboost_v1']**2.
            
        return catalog
    
    def remove_columns(self, columns = ['a','b']):
        
        cl_tab = Table(self.cl.galcat)
        
        cl_tab.remove_columns(columns)
        
        data = clmm.GCData(cl_tab)
        
        self.cl = clmm.GalaxyCluster('Stack', 0, 0, self.cl.z, data)
        
        return self.cl
        
        
    