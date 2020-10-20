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


def add_random_ellipticity(cl):

    r"""
    Parameters:
    ----------
    cl : GalaxyCluster object
        .compute_tangential_and_cross_component method is not applied
        
    Returns:
    -------
    cl : GalaxyCluster object
        with random ellipticity rotation
     """
    e1, e2 = cl.galcat['e1'], cl.galcat['e2']
    
    e = e1 + 1j * e2
    
    size = len(cl.galcat['e1'])
    
    angle = np.random.random(size) * 2 * np.pi
    
    cos_random = np.cos(angle)
    
    sin_random = np.sin(angle)

    rotation = ( cos_random + 1j * sin_random )
    
    e_tot = e*rotation

    cl.galcat['e1'] = e_tot.real
    
    cl.galcat['e2'] = e_tot.imag
    
    return cl


def random_rotate_ellipticity(cl):

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

    cl.galcat['e1'] = e.real
    
    cl.galcat['e2'] = e.imag
    
    return cl

def _randomize_redshift(cl):
    

    r"""
    Parameters:
    ----------
    cl : GalaxyCluster object
        .compute_tangential_and_cross_component method is not applied
        
    Returns:
    -------
    cl : GalaxyCluster object
        with randomized redshift using pzpdf.

     """
    
    if cl.galcat['pzsigma'] not in cl:
        
        print('no redshift error in your catalog')
        
        return cl
    
    else :

        size = len(cl.galcat['z'])
        
        probability = cl.galcat['pdfz']/np.sum(cl.galcat['pdfz'])
        
        zaxis = cl.galcat['pzbins']
        
        z_randomized = np.random.choice(zaxis, size , p=probablilty)
        
        cl.galcat['z'] = z_randomized
    
    return cl

class Statistics():

    def __init__(self, n_random_variable):
        
        X_label = ['Signal(R_' + f'{i})' for i in range(n_random_variable)]
        X_label = tuple(X_label)
        self.X_label = X_label
        
        DataType = ['f4' for i in range(n_random_variable)]
        DataType = tuple(DataType)
            
        self.X = Table(names = X_label, dtype = DataType)
        self.n_random_variable = n_random_variable
        self.real = 0
        
    def _add_realization(self, x_new):
    
        r"""
        add row for each new realization of random variable
        """
        self.real += 1
        self.X.add_row(tuple(x_new))
        
    def mean(self):
    
        average_signal = []
        
        for x_label in self.X_label : 
            
            average_signal.append(np.mean(self.X[x_label]))
    
        return average_signal
        
        
    def covariance(self):
        
        cov = np.zeros((self.n_random_variable,self.n_random_variable))
        
        for i, x_label in enumerate(self.X_label) : 
            
            for j, y_label in enumerate(self.X_label) :
                
                x = self.X[x_label]
                
                y = self.X[y_label]
                
                cov[i,j] = np.mean( (x-np.mean(x)) * (y - np.mean(y)) )
                
        return cov


