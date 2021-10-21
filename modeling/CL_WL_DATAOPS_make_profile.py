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
import random

import glob
from scipy import interpolate

def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] for i in range(wanted_parts) ]

def mean_value(profile = 1, 
               r_in = '1',
               gt_in = '1', 
               gx_in = '1',
               r_out = '1',
               gt_out = '1',
               gx_out = '1',
               weight = '1',):
    
    n_bins = len(profile[r_in][0])
    
    gt_w, gx_w, r_w, w = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)

    for p in profile:
        
        prob = p[weight]
        
        if weight == None: prob = np.array([1 if pw > 0 else 0 for pw in probc])
        
        gt_w = gt_w + p[gt_in]*prob
        gx_w = gx_w + p[gx_in]*prob
        r_w = r_w + p[r_in]*prob
        w = w + prob
    
    return gt_w/w, gx_w/w, r_w/w
        
def stacked_profile(profile = 1,
                    r_in = '1',
                    gt_in = '1', gx_in = '1',
                    r_out = '1',
                    gt_out = '1', gx_out = '1',
                    weight = '1',
                    z_name = '1', z_err = '1',
                    obs_name = '1', obs_err = '1',
                    Z_bin = 1, Obs_bin = 1, add_columns_to_bin = []):
    
    colname = ['z_mean','obs_mean','obs_rms','radius','gt','gx', 'gt_individual', 'radius_individual','n_stack','cluster_id','z_individual', 'obs_individual', 'z_bin', 'obs_bin']
    
    colname = colname + add_columns_to_bin
    
    data = {name : [] for name in colname}
    
    for z_bin in Z_bin:
        
        condition_z = (profile[z_name] < z_bin[1])*(profile[z_name] > z_bin[0])
        
        for obs_bin in Obs_bin:
            
            condition = condition_z * (profile[obs_name] < obs_bin[1]) * (profile[obs_name] > obs_bin[0])
            #condition = [c[0] for c in condition]
            p = profile[condition]
            if len(p) == 0: continue
            obs_mean, obs_rms = np.average(p[obs_name]), np.std(p[obs_name])/np.sqrt(len(p))
            z_mean = np.average(p[z_name])
                
            gt, gx, r = mean_value(profile = p, 
               r_in = r_in,
               gt_in = gt_in, 
               gx_in = gx_in,
               r_out = r_out,
               gt_out = gt_out,
               gx_out = gx_out,
               weight = weight)
            n = len(p)
            gt_individual = p[gt_in]
            radius_individual = p[r_in]
            array = [z_mean, obs_mean, obs_rms, r, gt, gx, gt_individual, radius_individual, n, p['cluster_id'], p[z_name], p[obs_name], z_bin, obs_bin]
            
            array = array + [p[name_add] for name_add in add_columns_to_bin]
                    
            for i, name in enumerate(colname):
                data[name].append(array[i])
                
    data = Table(data)
    return data

def bootstrap_covariance(profile = 1,
                    r_in = '1',
                    gt_in = '1', gx_in = '1',
                    r_out = '1',
                    gt_out = '1', gx_out = '1',
                    weight = '1',
                    z_name = '1', obs_name = '1',
                    n_boot = 1,
                    Z_bin = 1, Obs_bin = 1):
    
    colname = ['z_mean','obs_mean','obs_rms', 'cov_t', 'cov_x', 'gt_boot', 'gx_boot', 'gt_err', 'gx_err']
    
    data = {name : [] for name in colname}
    
    for z_bin in Z_bin:
        
        condition_z = (profile[z_name] < z_bin[1])*(profile[z_name] > z_bin[0])
        
        for obs_bin in Obs_bin:
            
            condition = condition_z*(profile[obs_name] < obs_bin[1])*(profile[obs_name] > obs_bin[0])
            
            #condition = [c[0] for c in condition]
            
            p = profile[condition]
            
            if len(p) == 0: continue
                
            obs_mean, obs_rms = np.mean(p[obs_name]), np.std(p[obs_name])
            z_mean = np.mean(p[z_name])
            index_id = np.arange(len(profile))
            cluster_id = profile['cluster_id']
            f = interpolate.interp1d(cluster_id, index_id)
            gt, gx = [], []
            
            for n in range(n_boot):
                
                cluster_id_boot = np.random.choice(p['cluster_id'], len(p))
                index_boot = f(cluster_id_boot)
                index_boot = np.array([int(index_) for index_ in index_boot])
                profile_boot = profile[index_boot]
                
                gt_boot, gx_boot, r_boot = mean_value(profile = profile_boot, 
                                       r_in = r_in,
                                       gt_in = gt_in, 
                                       gx_in = gx_in,
                                       r_out = r_out,
                                       gt_out = gt_out,
                                       gx_out = gx_out,
                                       weight = weight)

                gt.append(np.array(gt_boot)), gx.append(np.array(gx_boot))

            gt, gx = np.array(gt), np.array(gx)
            Xt, Xx = np.stack((gt.astype(float)), axis = 1), np.stack((gx.astype(float)), axis = 1)
            cov_t, cov_x = np.cov(Xt)*(len(profile_boot)-1)/len(profile_boot), np.cov(Xx)
            array = [z_mean, obs_mean, obs_rms, cov_t, cov_x, gt, gx, np.sqrt(cov_t.diagonal()), np.sqrt(cov_x.diagonal())]
            
            for i, name in enumerate(colname):
                data[name].append(array[i])
                 
    data = Table(data)
    
    return data

def bootstrap_maxi_covariance(profile = 1,
                    r_in = '1',
                    gt_in = '1', gx_in = '1',
                    r_out = '1',
                    gt_out = '1', gx_out = '1',
                    weight = '1',
                    z_name = '1', obs_name = '1',
                    n_boot = 1,
                    Z_bin = 1, Obs_bin = 1):
    
    colname = ['z_mean','obs_mean','obs_rms', 'cov_t', 'cov_x', 'gt_boot', 'gx_boot', 'gt_err', 'gx_err']
    
    data = {name : [] for name in colname}
    
    index_individual_clusters = profile['cluster_id']
    
    index_id_standard = np.arange(len(profile))
    
    DS_boot = []
    
    for n_bootstrap in range(n_boot):
        
        s = []
        
        index_standard_bootstrap = np.random.choice(index_id_standard, len(index_id_standard))
        
        mixed_profile = profile[index_standard_bootstrap]
        
        profile_boot = stacked_profile(profile = mixed_profile,
                    r_in = r_in,
                    gt_in = gt_in, gx_in = gx_in,
                    r_out = r_out,
                    gt_out = gt_out, gx_out = gt_out,
                    weight = weight,
                    z_name = z_name,
                    obs_name = obs_name, 
                    Z_bin = Z_bin, Obs_bin = Obs_bin)
        
        for p in profile_boot:
            
            s.extend(list(np.array(p['gt'])))
            
        DS_boot.append(np.array(s))
        
    DS_boot = np.array(DS_boot)
    DS_boot = np.stack((DS_boot.astype(float)), axis = 1)
        
    return DS_boot, np.cov(np.array(DS_boot))

def jackknife_maxi_covariance(profile = 1,
                    r_in = '1',
                    gt_in = '1', gx_in = '1',
                    r_out = '1',
                    gt_out = '1', gx_out = '1',
                    weight = '1',
                    z_name = '1', obs_name = '1',
                    n_jack = 1,
                    ra = 'ra', dec = 'dec',
                    Z_bin = 1, Obs_bin = 1):
    
    n_jack_ra = round(np.sqrt(n_jack))
    
    n_jack_dec = n_jack_ra
    
    ra_max, ra_min = np.max(profile[ra]), np.min(profile[ra])
    
    dec_max, dec_min = np.max(profile[dec]), np.min(profile[dec])
    
    ra_corner = np.linspace(ra_min, ra_max, n_jack_ra + 1)
    
    dec_corner = np.linspace(dec_min, dec_max, n_jack_dec + 1)
    
    Ra_bin  = binning(ra_corner)
    
    Dec_bin  = binning(dec_corner)
    
    colname = ['z_mean','obs_mean','obs_rms', 'cov_t', 'cov_x', 'gt_boot', 'gx_boot', 'gt_err', 'gx_err', 'Hartlap']
    
    data = {name : [] for name in colname}
    
    DS_jackk = []
    
    for ra_bin in Ra_bin:
                
                for dec_bin in Dec_bin:
                    
                    real = []
                
                    mask_jacknife = (profile[ra] > ra_bin[0])*(profile[ra] < ra_bin[1])*(profile[dec] > dec_bin[0])*(profile[dec] < dec_bin[1])

                    profile_jacknife = profile[np.invert(mask_jacknife)]
                    
                    profile_stack_jackknife = stacked_profile(profile = profile_jacknife,
                                                        r_in = r_in,
                                                        gt_in = gt_in, gx_in = gx_in,
                                                        r_out = r_out,
                                                        gt_out = gt_out, gx_out = gt_out,
                                                        weight = weight,
                                                        z_name = z_name,
                                                        obs_name = obs_name, 
                                                        Z_bin = Z_bin, Obs_bin = Obs_bin)
        
                    for p in profile_stack_jackknife:

                        real.extend(list(np.array(p['gt'])))

                    DS_jackk.append(np.array(real))

    DS_jackk = np.array(DS_jackk)
    
    DS_jackk = np.stack((DS_jackk.astype(float)), axis = 1)
    
    return DS_jackk, ((n_jack-1)**2/n_jack)*np.cov(np.array(DS_jackk))


        
        
        
        
def jacknife_covariance(profile = 1,
                    r_in = '1',
                    gt_in = '1', gx_in = '1',
                    r_out = '1',
                    gt_out = '1', gx_out = '1',
                    weight = '1',
                    z_name = '1', obs_name = '1',
                    n_jack = 1,
                    ra = 'ra', dec = 'dec',
                    Z_bin = 1, Obs_bin = 1):
    
    n_jack_ra = round(np.sqrt(n_jack))
    
    n_jack_dec = n_jack_ra
    
    ra_max, ra_min = np.max(profile[ra]), np.min(profile[ra])
    
    dec_max, dec_min = np.max(profile[dec]), np.min(profile[dec])
    
    ra_corner = np.linspace(ra_min, ra_max, n_jack_ra + 1)
    
    dec_corner = np.linspace(dec_min, dec_max, n_jack_dec + 1)
    
    Ra_bin  = binning(ra_corner)
    
    Dec_bin  = binning(dec_corner)
    
    colname = ['z_mean','obs_mean','obs_rms', 'cov_t', 'cov_x', 'gt_boot', 'gx_boot', 'gt_err', 'gx_err', 'Hartlap']
    
    data = {name : [] for name in colname}
    
    for z_bin in Z_bin:
        
        condition_z = (profile[z_name] < z_bin[1])*(profile[z_name] > z_bin[0])
        
        for obs_bin in Obs_bin:
            
            condition = condition_z*(profile[obs_name] < obs_bin[1])*(profile[obs_name] > obs_bin[0])
            
            condition = [c[0] for c in condition]
            
            p = profile[condition]
            
            if len(p) == 0: continue
                
            obs_mean, obs_rms = np.mean(p[obs_name]), np.std(p[obs_name])
            z_mean = np.mean(p[z_name])
            gt_JK, gx_JK = [], []

            for ra_bin in Ra_bin:
                
                for dec_bin in Dec_bin:
                
                    mask_jacknife = (p[ra] > ra_bin[0])*(p[ra] < ra_bin[1])*(p[dec] > dec_bin[0])*(p[dec] < dec_bin[1])

                    profile_jacknife = p[np.invert(mask_jacknife)]

                    gt_jk, gx_jk, r_jk = mean_value(profile = profile_jacknife, 
                                           r_in = r_in,
                                           gt_in = gt_in, 
                                           gx_in = gx_in,
                                           r_out = r_out,
                                           gt_out = gt_out,
                                           gx_out = gx_out,
                                           weight = weight)
                
                    gt_JK.append(np.array(gt_jk))
                    gx_JK.append(np.array(gx_jk))
            
            gt, gx = np.array(gt_JK), np.array(gx_JK)
            Xt, Xx = np.stack((gt.astype(float)), axis = 1), np.stack((gx.astype(float)), axis = 1)
            cov_t, cov_x = np.cov(Xt, bias = False), np.cov(Xx, bias = False)
            cov_t, cov_x = ((n_jack-1)**2/n_jack)*cov_t, ((n_jack-1)**2/n_jack)*cov_x
            H = (n_jack - cov_t.shape[0] - 2)/(n_jack - 1)

            
            array = [z_mean, obs_mean, obs_rms, cov_t, cov_x, gt, gx, np.sqrt(cov_t.diagonal()), np.sqrt(cov_x.diagonal()), H]
            
            for i, name in enumerate(colname):
                data[name].append(array[i])
                 
    data = Table(data)
    
    return data

def sample_covariance(profile = 1,
                    r_in = '1',
                    gt_in = '1', gx_in = '1',
                    r_out = '1',
                    gt_out = '1', gx_out = '1',
                    weight = '1',
                    z_name = '1', obs_name = '1',
                    Z_bin = 1, Obs_bin = 1):

    colname = ['z_mean','obs_mean','obs_rms', 'cov_t', 'cov_x', 'gt_boot', 'gx_boot', 'gt_err', 'gx_err']
    
    data = {name : [] for name in colname}
    
    for z_bin in Z_bin:
        
        condition_z = (profile[z_name] < z_bin[1])*(profile[z_name] > z_bin[0])
        
        for obs_bin in Obs_bin:
            
            condition = condition_z*(profile[obs_name] < obs_bin[1])*(profile[obs_name] > obs_bin[0])
            
            condition = [c[0] for c in condition]
            
            p = profile[condition]
            
            if len(p) == 0: continue
                
            obs_mean, obs_rms = np.mean(p[obs_name]), np.std(p[obs_name])
            z_mean = np.mean(p[z_name])
            gt_individual, gx_individual, w_unit = [], [], []
            
            for i, p_individual in enumerate(p):
                
                gt_individual.append(np.array(p_individual[gt_in]))
                gx_individual.append(np.array(p_individual[gx_in]))
                w_unit.append(np.array([1 if w_ > 0 else 0 for w_ in p_individual[weight]]))
                
            p['w_unit'] = np.array(w_unit)
                
            gt_mean = np.sum(p[gt_in]*p['w_unit'], axis = 0)/np.sum(p['w_unit'], axis = 0)
            
            cov_diag_sample = np.sum((p[gt_in] - gt_mean)**2*p['w_unit'], axis = 0)/(np.sum(p['w_unit'], axis = 0) - 1)
            
            cov_diag_mean = cov_diag_sample/len(p)
            
            cov_t = np.zeros([len(cov_diag_mean),len(cov_diag_mean)])
            cov_x = np.zeros([len(cov_diag_mean),len(cov_diag_mean)])
            for i in range(len(cov_diag_mean)):
                cov_t[i,i] = cov_diag_mean[i]

            #gt, gx, w = np.array(gt_individual), np.array(gx_individual), np.array(weights_individual)
            #Xt, Xx = np.stack((gt.astype(float)), axis = 1), np.stack((gx.astype(float)), axis = 1)
            #cov_t, cov_x = np.cov(Xt, aweights =  None)/len(p), np.cov(Xx)/len(p)
            
            array = [z_mean, obs_mean, obs_rms, cov_t, cov_diag_mean, 1, 1, cov_diag_sample, cov_diag_sample]
            
            for i, name in enumerate(colname):
                data[name].append(array[i])
                 
    data = Table(data)
    
    return data

