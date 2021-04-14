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
    
import settings_analysis as settings

sys.path.append('/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/GalaxyClusterCatalogs/dc2_object_run2.2i_dr6/tract/')

cosmo = settings.cosmo

moo = clmm.Modeling(massdef = 'mean', delta_mdef = 200, halo_profile_model = 'nfw')

moo.set_cosmo(cosmo)

class MetaCatalog():
    
    r"""
    A class for creating stacked background galaxy catalog and deriving shear profile using stacking estimation.
    """
    
    def __init__(self, is_deltasigma = True, cosmo = 0):
        
        self.is_deltasigma = is_deltasigma
        
        self.cosmo = cosmo
            
            
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
        
        file_where_source = np.array(glob.glob(where_source + '/cluster_ID_*'))

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
        
        self.file_in_bin = np.array([where_source + '/cluster_ID_' + str(h_id) + '.pkl' for h_id in self.selected_halo_id])
        
        self.file_in_bin = self.file_in_bin[np.isin(self.file_in_bin, file_where_source)]
                
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

        print(f'n = there are {len(self.file_in_bin)} available catalogs')
        
        return 0
            
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
        
        e_err_HSM = []
        
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
            
            try:
            
                sigma_c_1_v1.extend(cl_source.galcat['<sigma_c-1>_v1']), sigma_c_1_flexzboost_v1.extend(cl_source.galcat['<sigma_c-1>_flexzboost_v1'])
            
                odds_v1.extend(cl_source.galcat['z_odds_photoz_v1']), odds_flexzboost_v1.extend(cl_source.galcat['z_odds_photoz_flexzboost_v1'])
                
            except:
                
                a = 1
                
            try:
                
                e_err_HSM.extend(cl_source.galcat['e_err_HSM'])
                
            except:
                
                a = 1
            
            
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
        try:
            
            t['<sigma_c-1>_v1'] = np.array(sigma_c_1_v1)
            t['z_odds_photoz_v1'] = np.array(odds_v1)
            t['<sigma_c-1>_flexzboost_v1'] = np.array(sigma_c_1_flexzboost_v1)
            t['z_odds_photoz_flexzboost_v1'] = np.array(odds_flexzboost_v1)
            
        except: 
            
            a = 1
            
        try:
            
            t['e_err_HSM'] = np.array(e_err_HSM)
            
        except: 
            
            a = 1
        
        data = clmm.GCData(t)
        
        self.cl = clmm.GalaxyCluster('Stack', 0, 0, np.mean(cluster_z), data)
        
        self.cl.n_stacked_catalogs = len(self.file_in_bin)
        
        self.cl.n_galaxy = len(self.cl.galcat['halo_id'])
        
        
    def make_GalaxyCluster_catalog_test(self, shape_component1 = 'shear1', shape_component2 = 'shear2', tan_component = 'et', cross_component = 'ex', column_to_extract = 1):
        
        """
        Make GalaxyCluster catalogs with corresponding r, phi, ra, dec, e1, e2, et, ex, z_gal, sigma_c
        galaxy related quantities
        
        Returns:
        -------
        cl : GalaxyCluster
            Galaxy cluster catalog stacked along all individual clusters
        """ 
        
        halo_id = []
        
        column_to_extract = column_to_extract + ['r']
        
        cat = {name:[] for name in column_to_extract}

        for i, file in enumerate(self.file_in_bin):

            cl_source = load(file)
            
            try:

                cl_source.galcat['g1'], cl_source.galcat['g2'] = cl_source.galcat['e1'], cl_source.galcat['e2']

                cl_source.galcat['e1'], cl_source.galcat['e2'] = clmm.utils.compute_lensed_ellipticity(cl_source.galcat['e1_true'], cl_source.galcat['e2_true'], cl_source.galcat['shear1'], cl_source.galcat['shear2'], cl_source.galcat['kappa'])
                
            except : a = 1

            cl_source.compute_tangential_and_cross_components(geometry = "flat",
                                                              shape_component1 = shape_component1,
                                                              shape_component2 = shape_component2,
                                                              tan_component = tan_component, cross_component = cross_component, 
                                                              is_deltasigma = False)

            ut._add_distance_to_center(cl_source, self.cosmo)
            
            try: cl_source.galcat['sigma_c_1'] = moo.eval_sigma_crit(cl_source.z, cl_source.galcat['z'])**(-1) except: a = 1
                    
            halo_id.append()
            
            
        for name in colname:
            
            cat[name].extend(cl_source.galcat[name])
        
        data = clmm.GCData(t)
        
        self.cl = clmm.GalaxyCluster('Stack', 0, 0, 0, data)
        
        self.cl.n_stacked_catalogs = len(self.file_in_bin)
    
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
    
    def compute_weights(self, photoz = 1, err_measurement = 1, catalog = 1):
        
        if err_measurement == True:
            
            
            
            w_shape = (np.std(catalog.galcat['et'])**2 + (catalog.galcat['e_err_HSM']**2)/2)**(-1)
            
            print(w_shape)
            
        else:
            
            w_shape = 1
        
        if photoz == False:
        
            catalog.galcat['w_ls_true'] = w_shape * catalog.galcat['sigma_c']**(-2.)
        
        elif photoz == 'BPZ': 
            
            catalog.galcat['w_ls_v1'] = w_shape * catalog.galcat['<sigma_c-1>_v1']**2.
            
        elif photoz == 'fleXZboost':
            
            catalog.galcat['w_ls_flexzboost_v1'] = w_shape * catalog.galcat['<sigma_c-1>_flexzboost_v1']**2.
            
        return catalog
    
    def remove_columns(self, columns = ['a','b']):
        
        cl_tab = Table(self.cl.galcat)
        
        cl_tab.remove_columns(columns)
        
        data = clmm.GCData(cl_tab)
        
        self.cl = clmm.GalaxyCluster('Stack', 0, 0, self.cl.z, data)
        
        return self.cl
        
        
    