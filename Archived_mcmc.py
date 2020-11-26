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
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import modeling as model
import emcee

sys.path.append('/pbs/throng/lsst/users/cpayerne/GitForThesis/DC2Analysis')

from statistics_ import Statistics

class MCMC():
    r"""
    Motivations : fit cluster paramaters using MCMC approach complementary to curve_fit.

    Attributes:
    ---------
    n_parameters : int
        number of paramaters to fit
    n_walkers : int
        number of walkers for mcmc
    n_step : number of steps for each walkers
    """
    
    
    def __init__(self,xdata_dimension = 0,n_parameters = 0, n_walkers = 0, n_step = 0):
        
        self.xdata_dimension = xdata_dimension
        self.n_parameters = n_parameters
        self.n_walkers = n_walkers
        self.n_step = n_step
        
        self.ydata = None
        self.xdata = None
        self.covariance_matrix = None
        
        self.model = None
        
        self.sampler = None
        self.sample = None
        
    def _set_ydata(self, ydata):
        
        r"""
        selecting y_data
        """
        
        self.ydata = ydata
        
    def _set_xdata(self, xdata):
        
        r"""
        selecting x_data as a two d array with all values given each axis
        """
        
        self.xdata = xdata
        

    def _set_covariance_matrix(self, cov):
        
        r"""
            selecting covariance matrix of y_data
        """
    
        self.covariance_matrix = cov
        
    def _set_model(self, model):
        
        self.model = model
        
    def predicted_y(self, x, p):
        
        return self.model(x, p)
    
    def lnlike(self, p):
        
        "Gaussian Likelyhood"
        
        y_predict = np.array(self.model(self.xdata, p))
        
        delta = y_predict - self.ydata
        
        inv_cov = np.linalg.inv(self.covariance_matrix)
        
        return -0.5*np.sum(delta * inv_cov.dot(delta))
        
    def _set_lnprior(self, lnprior):
        
        "prior to be implemented as follow"
        
        self.lnprior = lnprior
    
    def lnprob(self, p):
        
        lp = self.lnprior(p)
        
        if not np.isfinite(lp):
            
            return -np.inf
        
        return lp + self.lnlike(p)
    
    def _set_initial_condition(self, p0, sigma):
        
        self.p0 = [(p0 + 3 * sigma * np.random.randn(self.n_parameters)).tolist() for i in range(self.n_walkers)]
        
    def run_MCMC(self):
        
        self.sampler = emcee.EnsembleSampler(self.n_walkers,self.n_parameters,self.lnprob)
        
        pos, prob, state = self.sampler.run_mcmc(self.p0, self.n_step,progress=True)
        
        self.acceptance_fraction = self.sampler.acceptance_fraction
        
    def _discard(self, tau = 0):
        
        tau = tau*np.array([1 for i in range(self.n_parameters)])
        
        burnin, thin = int(2 * np.max(tau)), int(0.5 * np.min(tau))
        
        self.samples = self.sampler.get_chain(discard=burnin, flat=True, thin=thin)
        
        
    def fit_symmetric(self):
        
        Stat = Statistics(self.n_parameters)
        
        for walk in self.samples: Stat._add_realization(np.array(walk))
            
        Stat.mean(), Stat.covariance()
        
        self.mean, self.covariance, self.error = Stat.mean, Stat.covariance, np.sqrt(Stat.covariance.diagonal())
        
        
    def fit_MCMC(self):
        
        mean, error = [], []
        
        for i in range(self.n_parameters):
            
            mcmc = np.percentile(self.samples[:, i], [16, 50, 84])
            
            q = np.diff(mcmc)
            
            mean.append(mcmc[1]), error.append([q[0],q[1]])
                        
        self.mean_MCMC, self.error_MCMC = np.array(mean), np.array(error)
