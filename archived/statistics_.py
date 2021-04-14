import numpy as np
from astropy.table import Table
import math as math

class Statistics():
    
    r"""
    
    A class for statisitcal analysis of data (computes mean and covariance matrices)
    
    """

    def __init__(self, n_random_variable):
        
        self.n_random_variable = n_random_variable
        
        DataType = tuple(['f4' for i in range(self.n_random_variable)])
        
        self.X_label = tuple(['X_' + f'{i}' for i in range(self.n_random_variable)])
            
        self.X = Table(names = self.X_label, dtype = DataType)
        
        self.realization = 0
        
    def _add_realization(self, x_new):
    
        r"""
        
        add row for each new realization of random variable
        
        
        """
        
        self.realization += 1
        
        self.X.add_row(tuple(x_new))
        
    def mean(self):
        
        r"""
        
        computes the mean of each random variable X_i as an attribute 
        
        """
    
        mean = []
        
        for x_label in self.X_label: 
            
            mean.append(np.nanmean(self.X[x_label]))
    
        self.mean = np.array(mean)
        
    def estimate_covariance(self):
        
        r"""
        
        computes the covariance matrix of the random variables X_i as an attribute
        
        """
        
        self.X_label = self.X.colnames
        
        cov_matrix = np.zeros((len(self.X_label),len(self.X_label)))
        
        for i, x_label in enumerate(self.X_label) : 
            
            for j, y_label in enumerate(self.X_label) :
                
                if j < i:
                    
                    continue
                
                x, y = self.X[x_label], self.X[y_label]
                
                mux, muy = np.mean(x), np.mean(y)
                
                cov_matrix[i,j] = np.sum((x - mux) * (y - muy))

                cov_matrix[j,i] = cov_matrix[i,j]
                
        self.covariance_matrix = cov_matrix
        
    def estimate_correlation(self):
        
        cov_matrix = self.covariance_matrix
        
        corr_matrix = np.zeros((len(self.X_label),len(self.X_label)))
        
        for i, x_label in enumerate(self.X_label) : 
            
            for j, y_label in enumerate(self.X_label) :
                
                if j < i:
                    
                    continue
                    
                corr_matrix[i,j] = cov_matrix[i,j]/np.sqrt(cov_matrix[i,i]*cov_matrix[j,j])
                
                corr_matrix[j,i] = corr_matrix[i,j]
                
        self.correlation_matrix = corr_matrix
