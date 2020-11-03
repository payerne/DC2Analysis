import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/CLMM/examples/support')
try: import clmm
except:
    import notebook_install
    notebook_install.install_clmm_pipeline(upgrade=False)
    import clmm
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from sampler import fitters
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy.optimize import curve_fit
import cluster_toolkit as ct
import GCRCatalogs
import os
import fnmatch
import scipy
import itertools
import clmm.polaraveraging as pa
import math


def content()
    print('reduced tangential shear')
    print('rho_c')
    print('delta_c')

def kappa_u(x):
    
    # intermediary function
    
    def inf(x):
        
        first = 1/(x**2  - 1)
        second = 1 - (2./np.sqrt(1-x**2))*np.arctanh(np.sqrt((1-x)/(1 + x)))
                                                  
        return 2*first*second
    
    def equal(x):
        
        first = 2./3
        
        return first
    
    def sup(x):
        
        first = 1/(x**2 - 1)
        second = 1 - (2./np.sqrt(x**2-1))*np.arctan(np.sqrt((x - 1)/(x + 1)))
        
        return 2*first*second
        
        
    if x > 1:
        
        return sup(x)
    
    if x == 1:
        
        return equal(x)
    
    if x < 1:
        
        return inf(x)

    
def shear_u(x):
    
    def ginf(x):
        
        racine = np.sqrt((1-x)/(1 + x))
        
        first = 8.*np.arctanh(racine)/(x**2*np.sqrt(1 - x**2))
        
        second = (4./x**2)*np.log(x/2)
        
        third = -2./(x**2 - 1)
        
        fourth = 4.*np.arctanh(racine)/((x**2 - 1)*np.sqrt(1 - x**2))
        
        return float(first + second + third + fourth)
    
    def gequal(x):
        
        first = 10./3 + 4*np.ln(1./2)
        
        return float(first)
    
    def gsup(x):
        
        racine = np.sqrt((x - 1)/(1 + x))
        
        first = 8.*np.arctan(racine)/(x**2*np.sqrt(x**2 - 1))
        
        second = (4./x**2)*np.log(x/2)
        
        third = -2./(x**2 - 1)
        
        fourth = 4.*np.arctan(racine)/((x**2 - 1)**(3./2))
        
        return float(first + second + third + fourth)
        
        
    if x > 1:
        
        return gsup(x)
    
    if x < 1:
        
        return ginf(x)
    
    if x == 1:
        
        return gequal(x)
    
def rho_c(cosmo, cluster_z):
    
    r = cosmo.critical_density(cluster_z).to(u.Msun / u.Mpc**3).value
    
    return cosmo.critical_density(cluster_z).to(u.Msun / u.Mpc**3).value


def delta_c(c, cosmo, z_cluster):
    
    return cosmo.Om(z_cluster)*(200./3)*c**3/(np.log(1 + c) - c/(1 + c))


def r200(M200, cosmo, z):

    Omega_m = cosmo.Om(z)
    
    first = Omega_m*rho_c(cosmo, z)*800*np.pi/3
    
    return (M200/first)**(1/3)

def rs(r200, c):

    return r200/c


def critical_density(cosmo, cluster_z, source_z): #Msun/Mpc
    
    
    first = cosmo.angular_diameter_distance(cluster_z).to(u.Mpc)
    
    second = cosmo.angular_diameter_distance(source_z).to(u.Mpc)
    
    third = cosmo.angular_diameter_distance_z1z2(cluster_z, source_z).to(u.Mpc)
    
    fourth = second/(first*third)
    
    G = const.G.to(u.Mpc**3 / (u.Msun * u.year**2))
    
    c = const.c.to(u.Mpc / u.year)
    
    fifth = c**2/(np.pi*4*G)
    
    return fifth*fourth


def coeff(M200, c, cosmo, z_cluster, z_source):
    
    rho_crit = rho_c(cosmo, z_cluster)
    
    r_200 = r200(M200, cosmo, z_cluster)
    
    r_s = rs(r_200,c)
    
    sigma_crit = critical_density(cosmo, z_cluster, z_source).value
    
    first = r_s*rho_crit*delta_c(c, cosmo, z_cluster)/sigma_crit
    
    return first

def shear(x, M200, c, cosmo, z_cluster, z_source):
    
    alpha = coeff(M200, c, cosmo, z_cluster, z_source)
    
    return float(shear_u(x))*alpha


def kappa(x, M200, c, cosmo, z_cluster, z_source):
    
    alpha = coeff(M200, c, cosmo, z_cluster, z_source)
    
    return kappa_u(x)*alpha


def reduced_tangential_shear(r, M200, c, z_cluster, z_source, cosmo,):
    
    #returns the reduced tangential shear for a cluster with mass M200, concentration c, with redshift z_cluster, and considering galaxies at redshift z_source. cosmo is an astropy object.
    
    r_200 = r200(M200, cosmo, z_cluster)
    
    r_s = rs(r_200,c)
    
    x = r/r_s
    
    return shear(x, M200, c, cosmo, z_cluster, z_source)/(1 - kappa(x, M200, c, cosmo, z_cluster, z_source))



