import pickle, sys, os
import numpy as np
import iminuit
from iminuit import Minuit
from astropy.table import Table, join

sys.path.append('/pbs/throng/lsst/users/cpayerne/GitForThesis/DC2Analysis')

import make_profile as prf

sys.path.append('/pbs/throng/lsst/users/cpayerne/GitForThesis/DC2Analysis/modeling/')

import miscentering as mis
import two_halo_term as twoh
import mass_conversion as utils

sys.path.append('/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/p_RedMapper_clusters/paper_dc2_galaxy_cluster_mass/data_DS/Halo_mass/module_Run_wl_mass_richness_relation/')

import module_fit_weak_lensing_mass as wl
import module_fit_mass_richness_relation as m_lambda
import module_SkySim5000_mass_richness_relation as skysim5000

import clmm
import clmm.dataops
from clmm.dataops import compute_tangential_and_cross_components, make_radial_profile, make_bins
from clmm.galaxycluster import GalaxyCluster
import clmm.utils as u
from clmm import Cosmology
from clmm.support import mock_data as mock
import pyccl as ccl

def load(filename, **kwargs):

    with open(filename, 'rb') as fin:
        
        return pickle.load(fin, **kwargs)

base_name = '/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/p_RedMapper_clusters/paper_dc2_galaxy_cluster_mass/data_DS/'
RedMapper = load('/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/Galaxy_Cluster_Catalogs_details/cosmoDC2/RedMapper_galaxy_clusters.pkl')

print(RedMapper.colnames)
cosmodc2_true_shape_true_z = load(base_name + 'Richness_cosmodc2_flex_BPZ/individual_true_shape_true_z_all_bins.pkl')
cosmodc2_epsilon_shape_true_z = load(base_name + 'Richness_cosmodc2_flex_BPZ/individual_epsilon_shape_true_z_all_bins.pkl')
cosmodc2_epsilon_shape_flex_z = load(base_name + 'Richness_cosmodc2_flex_BPZ/individual_epsilon_shape_flex_z_all_bins.pkl')
cosmodc2_epsilon_shape_BPZ_z = load(base_name + 'Richness_cosmodc2_flex_BPZ/individual_epsilon_shape_BPZ_z_all_bins.pkl')
dc2_HSM = load(base_name + '/DC2_object/individual_HSM_epsilon_shape_BPZ_z_all_bins.pkl')
dc2_Metacal = load(base_name + '/DC2_object/individual_Metacal_shape_BPZ_z_all_bins.pkl')
SkySim5000 = load(base_name + 'Halo_mass/SkySim5000.pkl')

profile_to_fit = [SkySim5000]
name = ['wl_SkySim5000.pkl']

def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]

z_corner = [0.2, 0.3,0.4,0.5,0.6]
rich_corner = np.round(np.logspace(np.log10(20),np.log10(200), 4))
Z_bin = binning(z_corner)
Richness_bin = binning(rich_corner)

Obs_bin = Richness_bin
n_boot = 400

z_min = 0
z_max = 2
richness_min = 0
richness_max = 1e5
abundance_min = 2

for index, prof in enumerate(profile_to_fit):

        print('Estimating shear profile')

        profile_stack = prf.stacked_profile(profile = prof,
                    r_in = 'radius_av',
                    gt_in = 'gt_av', gx_in = 'gx_av',
                    r_out = 'radius',
                    gt_out = 'gt', gx_out = 'gx',
                    weight = 'norm_sum',
                    z_name = 'cluster_z',
                    obs_name = 'richness',
                    Z_bin = Z_bin, Obs_bin = Obs_bin)

        print(profile_stack['obs_mean'])

        mask = profile_stack['n_stack'] > 2

        profile_stack = profile_stack[mask]

        print('Estimating covariance matrix')

        covariance_stack = prf.sample_covariance(profile = prof,
                        r_in = 'radius_av',
                        gt_in = 'gt_av', gx_in = 'gx_av',
                        r_out = 'radius',
                        gt_out = 'gt', gx_out = 'gx',
                        weight = 'norm_sum',
                        #n_boot = 400,
                        z_name = 'cluster_z', obs_name = 'richness',
                        Z_bin = Z_bin, Obs_bin = Obs_bin)

        covariance_stack = covariance_stack[mask]

        print('Fit weak lensing mass')

        fit_mass = wl.fit(profile = profile_stack, covariance = covariance_stack)

        print('Fit mass richness relation')

        data_to_save = {'name' : name[index],
                        'Z_bin' : Z_bin,'Richness_bin' : Richness_bin,
                        'stacked_shear_profile' : profile_stack,
                        'stacked_covariance_matrix' : covariance_stack, 
                        'weak_lensing_mass' : fit_mass}

        



