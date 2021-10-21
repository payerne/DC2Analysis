import pyccl as ccl
import matplotlib.pyplot as plt
import numpy as np
import scipy, pickle
from scipy import stats
from scipy.integrate import quad,simps, dblquad
from scipy.stats import poisson
from scipy.stats import multivariate_normal
from scipy.special import erfc
from scipy import interpolate
from math import factorial, gamma
import pickle as pkl
from astropy.cosmology import FlatLambdaCDM

def convolution_Gaussian_Poissonian_u(x = 1, mu_gauss = 1, sigma_gauss = 1, Lambda_poiss = 1):
    
    rv = poisson(Lambda_poiss)
    
    n_max = mu_gauss + sigma_gauss * 10 + 10 * Lambda_poiss + Lambda_poiss
    
    n = np.arange(n_max)
    
    Poiss = rv.pmf(n)
    
    Gauss = multivariate_normal.pdf(x - n, mean=mu_gauss, cov=sigma_gauss**2)
        
    return np.sum(Gauss*np.array(Poiss))

def convolution_Gaussian_Poissonian(x_array = 1, mu_gauss = 1, sigma_gauss = 1, Lambda_poiss = 1):
    
    res = []
    
    for X in x_array:
        
        a = convolution_Gaussian_Poissonian_u(x = X, mu_gauss = mu_gauss, sigma_gauss = sigma_gauss, Lambda_poiss = Lambda_poiss)
        
        res.append(a)
        
    return np.array(res)

def P(x_array = 1, mu = 1, sigma = 1):
    
    cov = sigma**2
    
    K1 = (1./np.sqrt(2*np.pi*cov))
    
    K2 = np.sqrt(1/cov)
    
    K3 = np.sqrt(np.pi/2)
    
    K4 = mu*K2/np.sqrt(2)
    
    K = 1 - (K3*erfc(K4)/K2)*K1
    
    def poissonian(x = 1, mu = 1):
        
        rv = poisson(mu)
    
        return rv.pmf(x)
    
    def Gaussian(x = 1, mu = 1, sigma = 1):
        
        return multivariate_normal.pdf(x, mean=mu, cov=sigma**2)
    
    res = []
    
    u = np.linspace(0, 10*mu * 3*sigma, 100000)
    
    for x_ in x_array:
        
        def __integrand__(u):
            
            return poissonian(x = x_, mu = u)*Gaussian(x = u, mu = mu, sigma = sigma)
        
        y_array = __integrand__(u)
        
        #a = quad(__integrand__, 0, 1e3)[0]
        
        a = simps(y_array,u)

        res.append(a)
        
    return np.array(res)/K

def integral_special(a,b,c):
    
    A = 0.5*c**(-1-a/2)
    
    B = np.sqrt(c)*gamma(0.5+a/2)*scipy.special.hyp1f1((a+1)/2,0.5,b**2/(4*c))
    
    C = b*gamma(1 + a/2)*scipy.special.hyp1f1(1+a/2.,3./2.,b**2./(4*c))
    
    return A*(B-C)

def P_cluster_count_u(N = 1, mu = 1, cov = 1):
    
    r"""
    Build K
    """
    
    K1 = (1./np.sqrt(2*np.pi*cov))
    
    K2 = np.sqrt(1/cov)
    
    K3 = np.sqrt(np.pi/2)
    
    K4 = mu*K2/np.sqrt(2)
    
    K = 1 - (K3*erfc(K4)/K2)*K1
    
    r"""
    Build constant coefficient
    
    """
    
    coeff = K1*np.exp(-mu**2/(2*cov))/factorial(N)
    
    r"""
    
    Build special function
    
    """
    
    A = N
    
    B = 1 - mu/cov
    
    C = 1/(2*cov)
    
    special_integral = integral_special(A,B,C)
        
    return K**(-1)*coeff*special_integral

def P_cluster_count(N_array = 1, mu = 1, cov = 1):
    
    res = []
    
    for n in N_array:
        
        res.append(P_cluster_count_u(N = n, mu = mu, cov = cov))
        
    return np.array(res)
