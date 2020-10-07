import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/CLMM/examples/support')
try: import clmm
except:
    import notebook_install
    notebook_install.install_clmm_pipeline(upgrade=False)
    import clmm
import numpy as np
from astropy.table import Table

def concentration(m, z_cl):
    
    #return the concentration of a cluster of mass m at given redshift z_cl
    #Duffy (207)
    
    a , b, c = 10.14, - 0.081,  - 1.01
    m0 = 2 * 10**(12)
    
    return a * ( m/m0 )**b *( 1 + z_cl )**c


def Sigmoid(r, r_0, r_c):
    #modelling of the sigmoid
    A = 1.    
    return A/(1 + np.exp(-(r - r_0)/r_c))


def d(z):
    #modelling of the distance d of the attenuated region at redshift z
    return 0.96*z + 0.32


def  predict_reduced_tangential_shear_z_distrib(r, logm, cluster_z, z_gal, cosmo):
    #returns the predict reduced tangential shear at physical distance r from the cluster center of mass m
    #for a collection of galaxies at several redshift data['z_gal'] following Chang averaging
    
    m = 10**logm
    Ngals = int(len(z_gal))
    nbins = int(Ngals**(1/2))
    
    hist, bin_edges = np.histogram(z_gal, nbins)
    Delta = bin_edges[1] - bin_edges[0]
    
    bin_center = bin_edges + Delta/2
    bin_center = list(bin_center)
    bin_center.pop(nbins)
    z = bin_center
    gt_model = []
    
    c = concentration(m,cluster_z)
    for i,R in enumerate(r):
        shear = hist*clmm.predict_reduced_tangential_shear(R*cosmo.h,
                                                                     m*cosmo.h, c,
                                                                     cluster_z, z, cosmo,                                                          delta_mdef=200,
                                                                     halo_profile_model='nfw')  
        gt_model.append(np.mean(shear)/nbins)
        
    return gt_model


def predict_excess_surface_density(r, logm, cluster_z, z_gal, cosmo):
    #doesnot depend of galaxy redshifts
    m = 10**logm
    
    c = concentration(m,cluster_z)
    
    critSD = np.array(clmm.get_critical_surface_density(cosmo, cluster_z, z_gal))
    meancritSD_1 = np.mean(critSD**(-1))
    meancritSD_3 = np.mean(critSD**(-3))
    
    deltasigma = []
    
    for i, R in enumerate(r):
        
        surface_density_nfw = clmm.predict_surface_density(R*cosmo.h, m*cosmo.h, c, cluster_z, cosmo, delta_mdef=200,
                                   halo_profile_model='nfw')
        
        order_0 = clmm.predict_excess_surface_density(R*cosmo.h, m*cosmo.h, c, cluster_z, cosmo, delta_mdef=200,
                                   halo_profile_model='nfw')
        
        order_1 = order_0*surface_density_nfw*meancritSD_3/meancritSD_1
        
        deltasigma.append(order_0 + order_1)
        
    return deltasigma
