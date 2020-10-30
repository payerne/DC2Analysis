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


