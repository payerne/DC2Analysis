import pyccl as ccl
import matplotlib.pyplot as plt
import numpy as np
import scipy, pickle
from scipy import stats
from scipy.integrate import quad,simps, dblquad
from scipy import interpolate
import pickle as pkl
from itertools import combinations, chain
import pandas as pd
import healpy
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt

def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]

class covariance_matrix():
    
    r"""
    Class for the computation of covariance matrices for cluster abundance
    """
    
    def __init__(self, catalog = 1, cosmo = 1):
        
        self.name = 'Covariance matrix for cluster count cosmology'
        self.catalog = catalog
        self.cosmo = cosmo
        
    def compute_boostrap_covariance(self, proxy_colname = 1, redshift_colname = 1,
                                          proxy_corner = 1, z_corner = 1,
                                          n_boot = 100, fct_modify = 1):
        
        proxy, redshift = self.catalog[proxy_colname], self.catalog[redshift_colname]

        index = np.arange(len(proxy))

        data_boot = []

        for i in range(n_boot):

            index_bootstrap = np.random.choice(index, len(index))

            data, proxy_edges, z_edges = np.histogram2d(redshift[index_bootstrap], proxy[index_bootstrap],
                                                   bins=[z_corner, proxy_corner])

            data_boot.append(data.flatten())

        data_boot = np.array(data_boot)

        N = np.stack((data_boot.astype(float)), axis = 1)

        mean = np.mean(data_boot, axis = 0)

        cov_N = np.cov(N, bias = False)

        self.Bootstrap_covariance_matrix = cov_N

    def compute_jacknife_covariance_healpy(self, proxy_colname = '', redshift_colname = '',
                                z_corner = '', proxy_corner = '', ra_colname = '', dec_colname = '',
                                n_power = 32, N_delete = 1):
        
        proxy, redshift = self.catalog[proxy_colname], self.catalog[redshift_colname]
        
        ra, dec =  self.catalog[ra_colname], self.catalog[dec_colname]
        
        index = np.arange(len(proxy))

        healpix = healpy.ang2pix(2**n_power, ra, dec, nest=True, lonlat=True)
        
        healpix_list_unique = np.unique(healpix)
        
        healpix_combination_delete = list(combinations(healpix_list_unique, N_delete))

        data_jack = []

        for i, hp_list_delete in enumerate(healpix_combination_delete):

                mask_in_area = np.isin(healpix, hp_list_delete)

                mask_out_area = np.invert(mask_in_area)

                data, mass_edges, z_edges = np.histogram2d(redshift[mask_out_area], 
                                                           proxy[mask_out_area],
                                                           bins=[z_corner, proxy_corner])
                
                data_jack.append(data.flatten())

        data_jack = np.array(data_jack)

        N = np.stack((data_jack.astype(float)), axis = 1)
        
        n_jack = len(healpix_combination_delete)

        cov_N = (n_jack - 1) * np.cov(N, bias = False,ddof=0)

        coeff = (n_jack-N_delete)/(N_delete*n_jack)
        
        self.n_jack = n_jack

        self.Jackknife_covariance_matrix = cov_N * coeff
        
    def compute_super_sample_covariance(self, proxy_colname = '', redshift_colname = '',
                                z_corner = [], proxy_corner = [], catalogs_name=[], fct_modify = 1):
        
        data_list = []
        
        for cat_name in catalogs_name:
            
            cat_individual = pd.read_csv(cat_name ,sep=' ', skiprows=12, names=['M','z','dec','ra'])
            
            fct_modify(cat_individual)
            
            data_individual, proxy_edges, z_edges = np.histogram2d(cat_individual[redshift_colname],
                                                                   cat_individual[proxy_colname], 
                                                                  bins=[z_corner, proxy_corner])
            
            data_list.append(data_individual.flatten())
            
        data = np.array(data_list)
        
        self.data_all_catalog = data
    
        N = np.stack((data.astype(float)), axis = 1)

        mean = np.mean(N, axis = 0)

        cov_N = np.cov(N, bias = False)
        
        self.super_sample_covariance_matrix = cov_N
        
        return data, cov_N
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
r"""
    
    
    
    
    
    


def compute_jacknife_covariance(mass = 1, redshift = 1,
                                z_corner = 1, m_corner = 1, ra = 1, dec = 1,
                                n_jack = 100):
    
    index = np.arange(len(mass))
    
    n_jack_ra = round(np.sqrt(n_jack))
    
    n_jack_dec = n_jack_ra
    
    ra_max, ra_min = np.max(ra), np.min(ra)
    
    dec_max, dec_min = np.max(dec), np.min(dec)
    
    ra_corner = np.linspace(ra_min, ra_max, n_jack_ra + 1)
    
    dec_corner = np.linspace(dec_min, dec_max, n_jack_dec + 1)
    
    Ra_bin  = binning(ra_corner)
    
    Dec_bin  = binning(dec_corner)

    data_jack = []
    
    Alpha = (1-1/n_jack)
    
    for i, ra_bin in enumerate(Ra_bin):
        
        for j, dec_bin in enumerate(Dec_bin):
            
            mask = (ra > ra_bin[0])*(ra < ra_bin[1])*(dec > dec_bin[0])*(dec < dec_bin[1])
            
            mask = np.invert(mask)
        
            data, mass_edges, z_edges, im  = plt.hist2d(redshift[mask],np.log10(mass[mask]) ,
                                                   bins=[z_corner,np.log10(m_corner)], cmin=0);

            where_are_NaNs = np.isnan(data)

            data[where_are_NaNs] = 0

            data_jack.append(data.flatten())
    
    data_jack = np.array(data_jack)
    
    N = np.stack((data_jack.astype(float)), axis = 1)
    
    mean = np.mean(data_jack/Alpha, axis = 0)
    
    cov_N = np.cov(N, bias = False)
    
    H = (n_jack - 1 - 2)/(n_jack - 1)
    
    coeff = ((n_jack-1)**2/n_jack)
            
    return coeff*cov_N/H, mean, data_jack


def compute_sample_variance(mass = 1, redshift = 1,
                                z_corner = 1, m_corner = 1, ra = 1, dec = 1,
                                n_region = 100):
    
    
    n_region_ra = round(np.sqrt(n_region))
    
    n_region_dec = n_region_ra
    
    ra_max, ra_min = np.max(ra), np.min(ra)
    
    dec_max, dec_min = np.max(dec), np.min(dec)
    
    ra_corner = np.linspace(ra_min, ra_max, n_region_ra + 1)
    
    dec_corner = np.linspace(dec_min, dec_max, n_region_dec + 1)
    
    Ra_bin  = binning(ra_corner)
    
    Dec_bin  = binning(dec_corner)

    data_region = []
    
    dOmega_list = []
    
    for i, ra_bin in enumerate(Ra_bin):
        
        for j, dec_bin in enumerate(Dec_bin):
            
            mask = (ra > ra_bin[0])*(ra < ra_bin[1])*(dec > dec_bin[0])*(dec < dec_bin[1])
            
            dOmega = (ra_bin[1]-ra_bin[0])*(np.pi/180)*(np.sin(dec_bin[1]*np.pi/180) - np.sin(dec_bin[0]*np.pi/180))
        
            data, mass_edges, z_edges, im  = plt.hist2d(redshift[mask],np.log10(mass[mask]) ,
                                                   bins=[z_corner,np.log10(m_corner)], cmin=0);
            
            dOmega_list.append(dOmega)

            where_are_NaNs = np.isnan(data)
            
            data = data

            data[where_are_NaNs] = 0

            data_region.append(data.flatten())
    
    data_region = np.array(data_region)
    
    N = np.stack((data_region.astype(float)), axis = 1)
    
    mean = np.mean(data_region, axis = 0)
    
    cov_N = np.cov(N, bias = False)
    
    coeff = (n_region)/(n_region - 1)
    
    return coeff*cov_N, mean, data_region, dOmega_list


def compute_sample_variance_healpy(mass = 1, redshift = 1,
                                z_corner = 1, m_corner = 1, ra = 1, dec = 1,
                                n_region = 10):
    
    
    n_region_ra = round(np.sqrt(n_region))
    
    n_region_dec = n_region_ra
    
    ra_max, ra_min = np.max(ra), np.min(ra)
    
    dec_max, dec_min = np.max(dec), np.min(dec)
    
    S0_deg2 = (ra_max-ra_min)*(dec_max-dec_min)/n_region
    
    ra_corner = np.linspace(ra_min, ra_max, n_region_ra + 1)
    
    dra = ra_corner[1]-ra_corner[0]
    
    ds = np.arcsin((np.pi/180)*S0_deg2/dra)*180/np.pi

    dec_corner = dec_min + np.arange(1000)*ds
    
    dec_corner = dec_corner[dec_corner <= dec_max]

    Ra_bin  = binning(ra_corner)
    
    Dec_bin  = binning(dec_corner)
    
    print(len(Dec_bin)*len(Ra_bin))

    data_region = []
    
    for i, ra_bin in enumerate(Ra_bin):
        
        for j, dec_bin in enumerate(Dec_bin):
            
            mask = (ra > ra_bin[0])*(ra < ra_bin[1])*(dec > dec_bin[0])*(dec < dec_bin[1])

            data, mass_edges, z_edges, im  = plt.hist2d(redshift[mask],np.log10(mass[mask]) ,
                                               bins=[z_corner,np.log10(m_corner)], cmin=0);

            where_are_NaNs = np.isnan(data)

            data = data

            data[where_are_NaNs] = 0

            data_region.append(data.flatten())
    
    data_region = np.array(data_region)
   
    return data_region

    

"""