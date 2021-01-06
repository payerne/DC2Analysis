import sys
import os
os.environ['CLMM_MODELING_BACKEND'] = 'nc' # here you may choose ccl or nc (NumCosmo)
sys.path.append('/pbs/throng/lsst/users/cpayerne/CLMM/examples/support')
try: import clmm
except:
    import notebook_install
    notebook_install.install_clmm_pipeline(upgrade=False)
    import clmm

from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.optimize import curve_fit
import numpy as np
from astropy.table import Table
import fnmatch
import pickle 

import clmm.polaraveraging as pa
import clmm.galaxycluster as gc
import clmm.modeling as modeling

from matplotlib import gridspec
import matplotlib.ticker as ticker
plt.style.use('classic')
import glob

sys.path.append('/pbs/throng/lsst/users/cpayerne/GitForThesis/DC2Analysis')

import stacking as stack
import utils as ut


class Perform_Stacking():
    
    def __init__(self, is_deltasigma = True, cosmo = 0):
        
        self.is_deltasigma = is_deltasigma
        
        self.cosmo = cosmo
        
    def _check_available_catalogs(self,bin_def = 'M_fof', z_bin = [1,1], obs_bin = [1,1], where_source = '1', r_lim = 1):
        
        self.z_bin = z_bin
        
        self.bin_def = 'M_fof'
        
        self.obs_bin = obs_bin
        
        self.where_source = where_source
        
        self.r_lim = r_lim
        
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
            
            if (len(cl_source.galcat['id']) == 0) or (not ut._is_complete(cl_source, self.r_lim, self.cosmo)): 
                
                self.mask_end.append(False)
                
            else: self.mask_end.append(True)
                
        self.file_in_bin = self.file_in_bin[self.mask_end]
                
        if len(self.file_in_bin) < 5 : raise ValueError("Not enough catalogs for stacking")
            
        else : print(f'n = {len(self.file_in_bin)} no-fiterered available catalogs')
        
    def make_binned_profile(self, r_low = 0, r_up = 1, n_bins = 0, method = 'evenlog10width'):
        
        self.r_low = r_low
        
        self.r_up = r_up
        
        self.n_bins = n_bins
        
        self.Shapenoise = False
        
        self.method = method
        
        self.shear_component1 = 'e1'
        
        self.shear_component2 = 'e2'
        
        pf_source = stack.Stacking( r_low = self.r_low,  r_up = self.r_up, n_bins = self.n_bins, cosmo = self.cosmo )
        
        pf_source._select_type( is_deltasigma = self.is_deltasigma ) 
        
        for i, file in enumerate(self.file_in_bin): 

            os.chdir(self.where_source)

            cl_source = pickle.load(open(file,'rb'))

            cl_source.galcat['id'] = np.array([i for i in range(len(cl_source.galcat['id']))])
            
            if self.Shapenoise == True : cl_source = ut._add_shapenoise(cl_source)

            cl_source.compute_tangential_and_cross_components(geometry = "flat",
                                                              shape_component1 = self.shear_component1,
                                                              shape_component2 = self.shear_component2,
                                                              tan_component = 'et', cross_component = 'ex', 
                                                              is_deltasigma = True, 
                                                              cosmo = pf_source.cosmo)

            bin_edges = pa.make_bins( pf_source.r_low , pf_source.r_up , pf_source.n_bins , method = self.method)

            profile_source = cl_source.make_binned_profile("radians", "Mpc", 
                                                           bins = bin_edges,
                                                           cosmo = pf_source.cosmo,
                                                           include_empty_bins = True,
                                                           gal_ids_in_bins = True)
            
            pf_source._add_cluster_redshift(cl_source)
            
            pf_source._add_background_galaxies(cl_source, profile_source)

            pf_source._estimate_individual_lensing_signal(cl_source, profile_source)
            
        if pf_source.n_stacked_cluster < 5:
            
            raise ValueError("Not enough catalogs for stacking")
            
        pf_source.MakeStackedProfile()
        
        r"""
        Sample Covariance matrix
        """

        pf_source._add_standard_deviation()
        
        pf_source._reshape_data()

        self.profile = pf_source.profile
        
        self.cov_t, self.cov_x = pf_source.cov_t, pf_source.cov_x
        
        self.galaxy_redshift = sum(pf_source.z_galaxy, [])
    
    def _check_average_inputs(self):
    
        self.mass, self.mass_rms = np.mean(self.mass_in_bin[self.mask_end]), np.std(self.mass_in_bin[self.mask_end])
        
        self.z, self.z_rms = np.mean(self.z_in_bin[self.mask_end]), np.std(self.z_in_bin[self.mask_end])
        
        self.rich, self.rich_rms = np.mean(self.richness_in_bin[self.mask_end]), np.std(self.richness_in_bin[self.mask_end]) 

    def plot_profile(self):

        def myyticks(x,pos):

            if x == 0: return "$0$"
            sign = x/abs(x)
            coeff = (x)/10**13

            return r"${:2.1f}$".format(coeff)

        def myxticks(x,pos):

            if x == 0: return "$0$"
            exponent = int(np.log10(abs(x)))
            sign = x/abs(x)
            coeff = (x/(10**exponent))

            return r"${:2.0f}$".format(x)

        ylabelup = r'$\Delta\Sigma_+$ ' +'$[$' + r'$\times 10^{13}$ ' + r'${\rm M}$' + r'$_\odot\;$'+ r'${\rm Mpc}$'+r'$^{-2}$'r'$]$'
        ylabeldown = r'$\Delta\Sigma_\times$ ' +'$[$' + r'$\times 10^{13}$ ' + r'${\rm M}$' + r'$_\odot\;$'+ r'${\rm Mpc}$'+r'$^{-2}$'r'$]$'
        xlabel = r'$R\ [$' + r'${\rm Mpc}$' + r'$]$'

        # Simple data to display in various forms

        fig = plt.figure(figsize = (10,8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
        ax0 = plt.subplot(gs[0])
        ax0.tick_params(axis='both', which='major', labelsize=16)
        ax0.set_ylabel(ylabelup, fontsize=16)
        #ax0.set_ylabel('r'$\rm{stacked}\ \Delta\Sigma_\times$)

        # log scale for axis Y of the first subplot
        ax0.set_yscale("log")
        ax0.set_xscale("log")
        ax0.set_xlim(self.r_low, self.r_up)
        #ax0.set_ylim(2*10**12, 1.5*10**14)

        line0 = ax0.errorbar(self.profile_image['radius'], self.profile_image['gt'],self.profile_image['gt_err'],
                             fmt='s',capsize = 7 ,ecolor = 'b',elinewidth=1.2, c='b',\
                         markeredgecolor='b',markerfacecolor='None', \
                             markeredgewidth=1.2, markersize = 10,
                             label=r'$\rm{cosmoDC2\_v1.1.4\_image}$')

        yticks = ax0.yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)
        ax0.yaxis.set_major_formatter(ticker.FuncFormatter(myyticks))
        ax0.legend(loc='best', frameon = False,
                  numpoints = 1, fontsize = 15)

        ax1 = plt.subplot(gs[1], sharex = ax0)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.set_xlabel(xlabel, fontsize=16)
        ax1.set_ylabel(ylabeldown, fontsize=16)

        ax1.plot(self.profile_image['radius'], 0*self.profile_image['radius'], '--k')
        ax1.legend(loc='best', frameon = False,
                  numpoints = 1, fontsize = 15)

        plt.setp(ax0.get_xticklabels(), visible=False)
        # remove last tick label for the second subplot

        yticks = ax1.yaxis.get_major_ticks()

        #yticks[-1].label1.set_visible(False)

        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(myyticks))
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(myxticks))
        #ax1.set_ylim(-1.5*10**13, 1.5*10**13)

        line0 = ax1.errorbar(self.profile_image['radius'], self.profile_image['gx'],self.profile_image['gx_err'],
                             fmt='s',capsize = 7 ,ecolor = 'b',elinewidth=1.2, c='b',\
                         markeredgecolor='b',markerfacecolor='None', \
                             markeredgewidth=1.2, markersize = 10,
                             label=r'$\rm{cosmoDC2\_v1.1.4\_image}$')

        # remove vertical gap between subplots
        plt.subplots_adjust(hspace=.0)
        #os.chdir('/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/GalaxyClusterCatalogs')
        #plt.savefig('DeltaSigma_Stacking_cosmodc2_dc2object.png', bbox_inches='tight', dpi=300)
        print(1)
        plt.show(block=True)

