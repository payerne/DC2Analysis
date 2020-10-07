import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/CLMM/examples/support')
try: import clmm
except:
    import notebook_install
    notebook_install.install_clmm_pipeline(upgrade=False)
    import clmm
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from sampler import fitters
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy.optimize import curve_fit
import cluster_toolkit as ct
import GCRCatalogs
import os
import fnmatch
import scipy
import math

import clmm.polaraveraging as pa
import clmm.utils as utils

import modelling as model
    

def shapenoise(cl_stack):
    
    es1 = cl_stack.galcat['e1_true']
    es2 = cl_stack.galcat['e2_true']
    gamma1 = cl_stack.galcat['shear1']
    gamma2 = cl_stack.galcat['shear2']
    kappa = cl_stack.galcat['kappa']
                
    e1_m, e2_m = utils.compute_lensed_ellipticity(es1, es2, gamma1, gamma2, kappa)
                
    cl_stack.galcat['e1'] = e1_m
    cl_stack.galcat['e2'] = e2_m
    
    return cl_stack

def make_gt_profile(cl_stack, up, down, n_bins, is_deltasigma, cosmo):
    #leg
    
    if (cl_stack != 1):
        
        cl_stack.compute_tangential_and_cross_components(geometry="flat", is_deltasigma = is_deltasigma, cosmo = cosmo)
        
        bin_edges = pa.make_bins(down, up , n_bins, method='evenlog10width')
        
        profile = cl_stack.make_binned_profile("radians", "Mpc", bins=bin_edges,cosmo=cosmo,include_empty_bins= True,gal_ids_in_bins=True)

        profile['gt'] = [np.nan if (math.isnan(i)) else i for i in profile['gt']]
        profile['gt_err'] = [np.nan if math.isnan(profile['gt'][i]) else err for i,err in enumerate(profile['gt_err'])]
        profile['radius'] = [np.nan if math.isnan(profile['gt'][i]) else radius for i,radius in enumerate(profile['radius'])]
        profile['cluster_z'] = cl_stack.z
        
        return profile
    
    else: 
        
        print('No profile available')
        
        return 1

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
        self.z_galaxy_list = []
        self.gt_list = []
        self.variance_gt_list = []
        self.radial_axis_list = []
        self.cosmo = cosmo
        self.profile = 0
        
        self.r_low = r_low
        self.r_up = r_up
        self.n_bins = n_bins
        
        self.n_stacked_gt = 0
        self.average_z = 0
        self.is_deltasigma = None
        
        
    def SelectType(self, is_deltasigma = True):
        
        """Indicates the type of profile to bin"""
        
        self.is_deltasigma = is_deltasigma
        
    
    def Add(self, cl_stack, Shapenoise = True):
        
        """add individual binned profile from individual galaxy catalog"""
        
        if self.is_deltasigma == None:
            
            raise ValueError(f"type of profile not defined yet ! available : tangential reduced shear or excess surface density")
    
        if Shapenoise == True:
            
            cl_stack = shapenoise(cl_stack)
        
        profile_stack = make_gt_profile(cl_stack, self.r_up, self.r_low, self.n_bins, self.is_deltasigma, self.cosmo)
    
        self.z_cluster_list.append(cl_stack.z)
        self.z_galaxy_list.extend(list(cl_stack.galcat['z']))
        self.gt_list.append(profile_stack['gt'])
        
        self.variance_gt_list.append(np.nan_to_num(profile_stack['gt_err']**2, nan = np.nan, posinf = np.nan, neginf = np.nan))
        self.radial_axis_list.append(profile_stack['radius'])
        
        self.n_stacked_gt += 1
        
        
    def MakeShearProfile(self, method):
        
        """Calculates the stacked profile from individual profiles"""
        
        if self.n_stacked_gt == 0:
            
            raise ValueError(f"Problem for makin{self.n_stacked_gt} loaded galaxy catalogs")
        elif self.n_stacked_gt == 1:
            raise ValueError(f"No loaded galaxy catalogs")
    
        gt = np.array(self.gt_list)
        variance_gt = np.array(self.variance_gt_list) 
        radius = np.array(self.radial_axis_list)
        
        if method == 'error weighted':
            
            w = np.nan_to_num(1/(np.array(variance_gt)), nan = np.nan, posinf = np.nan)
            
        elif method == 'classical': 
            
            w = np.nan_to_num(1*(gt/gt), nan = np.nan, posinf = np.nan)
            
        weight = w/np.nansum(w, axis = 0)
            
        profile = Table()
            
        gt_average = np.nansum(gt*weight, axis=0)
        gt_average_dispersion = np.sqrt(np.nansum(weight*(gt - gt_average)**2,axis=0))
        radius_average = np.nansum(weight*radius, axis=0)
        radius_average_dispersion = np.sqrt(np.nansum(weight*(radius - radius_average)**2,axis=0))

        profile['gt'] = gt_average
        profile['gt_err'] = gt_average_dispersion
        profile['radius'] = radius_average
        profile['radius_err'] = radius_average_dispersion
        
        self.average_z = np.mean(self.z_cluster_list)
        self.profile = profile
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
        

    