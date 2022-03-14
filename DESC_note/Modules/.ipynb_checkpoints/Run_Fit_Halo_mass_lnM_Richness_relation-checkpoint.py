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
#dc2_Metacal = load(base_name + '/DC2_object/individual_Metacal_epsilon_shape_BPZ_z_all_bins.pkl')
dc2_HSM = load(base_name + '/DC2_object/individual_HSM_epsilon_shape_BPZ_z_all_bins.pkl')
dc2_Metacal = load(base_name + '/DC2_object/individual_Metacal_shape_BPZ_z_all_bins.pkl')
SkySim5000 = load(base_name + 'Halo_mass/SkySim5000.pkl')

# profile_to_fit = [cosmodc2_epsilon_shape_true_z, cosmodc2_epsilon_shape_flex_z, cosmodc2_epsilon_shape_BPZ_z]
# name = ['cosmodc2_true_1h.pkl', 'cosmodc2_flex_1h.pkl', 'cosmodc2_BPZ_1h.pkl']

#p#rofile_to_fit = [dc2_Metacal]
#name = ['dc2_Metacal.pkl']
profile_to_fit = [dc2_HSM]
name = ['dc2_HSM.pkl']

# profile_to_fit = [cosmodc2_epsilon_shape_true_z]
# name = ['cosmodc2_wl_select_25_randomss.pkl']

fit_wl = True

simu_mass = [SkySim5000]
simu_name = ['SkySim5000_25_randomss.pkl']

fit_simu = False

def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]
#z_corner_long = np.linspace(0.2, 1, 6)
#rich_corner_long = np.logspace(np.log10(20.), np.log10(200),5)

# z_corner_short = [0.2, 0.4, 0.6, 0.8, 1.]
# rich_corner_short = [30, 60, 120, np.inf]

z_corner = [0.2, 0.3,0.4,0.5,0.6]
#z_corner = [0.2, 0.4, 0.6, 0.8, 1]
#rich_corner = np.round(np.logspace(np.log10(20),np.log10(200), 4))
#rich_corner = [20, 30, 70, 140, 250]
#rich_corner = [20., 33., 55., 91., 1000]

#rich_corner = [20, 30, 70, 140, 250]
#rich_corner = [20., 35., 55., 90., 1000]
#rich_corner = np.round(np.logspace(np.log10(20),np.log10(200), 6))
#rich_corner = [20, 24, 30, 40, np.inf ]
rich_corner = np.round(np.logspace(np.log10(20),np.log10(200), 4))
#rich_corner[-1] = np.inf
Z_bin = binning(z_corner)
Richness_bin = binning(rich_corner)

Obs_bin = Richness_bin
n_boot = 400

z_min = 0
z_max = 2
richness_min = 0
richness_max = 1e5
abundance_min = 2

# mask = np.isin(RedMapper['cluster_id'], profile_to_fit[0]['cluster_id'])

# RedMapper_cut = RedMapper[mask]

# simu_mass = Table(skysim5000.make_binned(match = skysim5000.match_1(base_catalog = RedMapper_cut), Z_bin = Z_bin, Richness_bin = Obs_bin))

# cluster_id = []

# for i, id_list in enumerate(simu_mass['cluster_id_RedMapper']):
#    
#    cluster_id.extend(list(id_list))

# cluster_id_selected = np.array(cluster_id)

# print(len(cluster_id_selected))
# print(simu_mass['richness'])

if fit_simu == True:

    sampler_emcee, fit_minuit, params = m_lambda.fit_M_lambda(file = simu_mass, logm = 'logm200', logm_err = 'logm200_err',
             z = 'z_mean', z_err = 'z_mean_err',
             richness = 'richness', richness_err = 'richness_err',
             abundance = 'n_stack',
             z_min = z_min, z_max = z_max,
             richness_min = richness_min, richness_max = richness_max,
             abundance_min = abundance_min)

    data_to_save = {'name' : simu_name[0],
                    'Z_bin' : Z_bin,'Richness_bin' : Richness_bin,
                    'stacked_shear_profile' : 1,
                    'stacked_covariance_matrix' : 1, 
                    'weak_lensing_mass' : simu_mass,
                   'sampler' : sampler_emcee,
                   'fit_minuit_m_lambda' : fit_minuit,
                   'parameters_to_fit' : params}

    with open('/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/p_RedMapper_clusters/paper_dc2_galaxy_cluster_mass/data_DS/Halo_mass/data_fit/M_Lambda_constraint_' + simu_name[0], 'wb') as file:

        pickle.dump(data_to_save , file)

if fit_wl == True:

    for index, prof in enumerate(profile_to_fit):
        
            #prof['redshift'] = prof['cluster_z']
            #prof_selected = prof[np.isin(prof['cluster_id'], cluster_id_selected)]
            #prof = prof_selected
            #print(prof['richness'])
            #print(RedMapper['redshift'])
            #profi = join(prof, RedMapper_cut[np.isin(RedMapper_cut['cluster_id'],prof['cluster_id'])], keys = 'cluster_id')
            
            #profi['richness'] = profi['richness_1']
            #print(prof.colnames)
            
            #gt_av = prof['gt_av']
            #R_T = prof['<R_T>']
            #a_calib = []
            #for i, gt in enumerate(gt_av):
            #    R = R_T[i]
            #    norm = prof['norm_sum'][i]
            #    a_calib.append(np.array([gt[j]/R[j] if norm[j] != 0 else 0 for j in range(len(gt#))]))
            #prof['gt_av'] = np.array(a_calib)

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
            
            #fit_mass['richness'] = simu_mass['richness']
            #fit_mass['richness_err'] = simu_mass['richness_err']
            #fit_mass['z_mean'] = simu_mass['z_mean']
            #fit_mass['z_mean_err'] = simu_mass['z_mean_err']

            print('Fit mass richness relation')

            sampler_emcee, fit_minuit, params = m_lambda.fit_M_lambda(file = fit_mass, logm = 'logm200', logm_err = 'logm200_err',
                     z = 'z_mean', z_err = 'z_mean_err',
                     richness = 'richness', richness_err = 'richness_err',
                     abundance = 'n_stack',
                     z_min = z_min, z_max = z_max,
                    richness_min = richness_min, richness_max = richness_max,
                    abundance_min = abundance_min)

            data_to_save = {'name' : name[index],
                            'Z_bin' : Z_bin,'Richness_bin' : Richness_bin,
                            'stacked_shear_profile' : profile_stack,
                            'stacked_covariance_matrix' : covariance_stack, 
                            'weak_lensing_mass' : fit_mass,
                           'sampler' : sampler_emcee,
                           'fit_minuit_m_lambda' : fit_minuit,
                           'parameters_to_fit' : params}

            with open('/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/p_RedMapper_clusters/paper_dc2_galaxy_cluster_mass/data_DS/Halo_mass/data_fit/M_Lambda_constraint_' + name[index], 'wb') as file:

                pickle.dump(data_to_save , file)



