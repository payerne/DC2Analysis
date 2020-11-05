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
import modeling as model


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
    
        mean = []
        
        for x_label in self.X_label: 
            
            mean.append(np.nanmean(self.X[x_label]))
    
        self.mean = np.array(mean)
        
        
    def covariance(self):
        
        r"""
        
        returns the covariance matrix of the random variable X_i
        
        """
        
        cov_matrix = np.zeros((self.n_random_variable,self.n_random_variable))
        
        for i, x_label in enumerate(self.X_label) : 
            
            for j, y_label in enumerate(self.X_label) :
                
                if j < i:
                    
                    continue
                
                x, y = self.X[x_label], self.X[y_label]
                
                mask = np.logical_not(np.isnan(x * y))
                
                x, y = x[mask], y[mask]
                
                n = len(x)
                
                if n > 1: cov_matrix[i,j] = np.sum( (x-np.mean(x)) * (y - np.mean(y)) ) / (n - 1)
                
                else: cov_matrix[i,j] = np.Nan
                    
                cov_matrix[j,i] = cov_matrix[i,j]
                
        self.covariance = cov_matrix


