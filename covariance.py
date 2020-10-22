import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/CLMM/examples/support')
try: import clmm
except:
    import notebook_install
    notebook_install.install_clmm_pipeline(upgrade=False)
    import clmm
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import numpy as np
from astropy.table import Table
import math

import clmm.polaraveraging as pa
import clmm.utils as utils

import modelling as model


def random_rotate_ellipticity(cl):

    r"""
    Parameters:
    ----------
    cl : GalaxyCluster object
        compute_tangential_and_cross_component method is not applied
        
    Returns:
    -------
    cl : GalaxyCluster object
        with random ellipticity rotation (vanishing the shear signal)
     """
    
    e1, e2 = cl.galcat['e1'], cl.galcat['e2']
    
    e = e1 + 1j * e2
    
    size = len(cl.galcat['e1'])
    
    angle = np.random.random(size) * 2 * np.pi
    
    cos_random = np.cos(angle)
    
    sin_random = np.sin(angle)

    rotation = ( cos_random + 1j * sin_random )
    
    e_tot = e * rotation

    cl.galcat['e1'] = e_tot.real
    
    cl.galcat['e2'] = e_tot.imag
    
    return cl


def add_random_ellipticity(cl):

    r"""
    Parameters:
    ----------
    cl : GalaxyCluster object
        .compute_tangential_and_cross_component method is not applied
        
    Returns:
    -------
    cl : GalaxyCluster object
        with random ellipticity contribution (with local constraint norm(epsilon) < 1)

     """
    e1, e2 = cl.galcat['e1'], cl.galcat['e2']
    
    e = e1 + 1j * e2
    
    size = len(cl.galcat['e1'])
    
    norm_e_random_max = 1 - np.sqrt(e1**2 + e2**2)
    
    norm_e_random = np.random.random(size) * norm_e_random_max
    
    angle = np.random.random(size) * 2 * np.pi
    
    cos_random = np.cos(angle)
    
    sin_random = np.sin(angle)

    e_small = norm_e_random * ( cos_random + 1j * sin_random )
    
    e_tot = e + e_small

    cl.galcat['e1'] = e_tot.real
    
    cl.galcat['e2'] = e_tot.imag
    
    return cl

def randomize_redshift(cl):
    

    r"""
    Parameters:
    ----------
    cl : GalaxyCluster object
        .compute_tangential_and_cross_component method is not applied
        
    Returns:
    -------
    cl : GalaxyCluster object
        with randomized redshift using pzpdf of cl.

     """
    
    if 'pzsigma' not in cl.galcat.keys():
        
        print('no implemented redshift error in your catalog !')
        
        return cl
    
    else :

        size = len(cl.galcat['z'])
        
        pz_pdf = np.array(cl.galcat['pdfz'])
        
        norm_pz_pdf = [np.sum(pz_pdf[i][:]) for i in range(size)]
        
        probability = cl.galcat['pdfz']/norm_pz_pdf
        
        zaxis = cl.galcat['pzbins']
        
        z_random = np.random.choice( zaxis , size , p = probablilty)
        
        cl.galcat['z'] = z_random
        
        return cl
    
    
def bootstrap(cl):
    
    r"""
Parameters:
----------
cl : GalaxyCluster object
    .compute_tangential_and_cross_component method is not applied

Returns:
-------
cl : GalaxyCluster object
    with bootstrap sampled catalog cl

 """
    
    size = len(cl.galcat['z'])
    
    index = [i for i in range(size)]
    
    index_random = np.random.choice(index, size)
    
    cl_bootstrap = cl
    
    for i, j in enumerate(index_random):
        
        cl_bootstrap.galcat[j] = cl.galcat[i]
    
    return cl_bootstrap
                      
    

class Statistics():
    
    r"""
    A class for statisitcal analysis of data
    """

    def __init__(self, n_random_variable):
        
        self.X_label = tuple(['X_' + f'{i}' for i in range(n_random_variable)])
 
        DataType = tuple(['f4' for i in range(n_random_variable)])
            
        self.X = Table(names = self.X_label, dtype = DataType)
        
        self.n_random_variable = n_random_variable
        
        self.realization = 0
        
    def _add_realization(self, x_new):
    
        r"""
        
        add row for each new realization of random variable
        
        
        """
        
        self.realization += 1
        
        self.X.add_row(tuple(x_new))
        
    def mean(self):
    
        mean_X = []
        
        for x_label in self.X_label : 
            
            mean_X.append(np.mean(self.X[x_label]))
    
        return mean_X
        
        
    def covariance(self):
        
        r"""
        
        returns the covariance matrix of the random variable X_i
        
        """
        
        cov_matrix = np.zeros((self.n_random_variable,self.n_random_variable))
        
        for i, x_label in enumerate(self.X_label) : 
            
            for j, y_label in enumerate(self.X_label) :
                
                x = self.X[x_label]
                
                y = self.X[y_label]
                
                cov_matrix[i,j] = np.sum( (x-np.mean(x)) * (y - np.mean(y)) ) / (self.realization - 1)
                
        return cov_matrix


