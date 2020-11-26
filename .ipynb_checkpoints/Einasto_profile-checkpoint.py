import numpy as np
from astropy import units as u
from astropy import constants as const
import math
from scipy.special import gamma, gammainc
import pyccl as ccl
from scipy.optimize import fsolve

class Modeling():

    def __init__(self, M200, concentration, A, cluster_z, mass_def, cosmo):
    
        self.mass_def = mass_def
        self.M200 = M200
        self.concentration = concentration
        self.cosmo = cosmo
        self.cluster_z = cluster_z
        self.a = A
        
        if self.mass_def == 'mean': self.alpha = self.cosmo.Om(self.cluster_z)
        else : self.alpha = 1.
        
    def rho_c(self,z):

        return self.cosmo.critical_density(z).to(u.Msun / u.Mpc**3).value
        
    def r200(self):

            return ((self.M200 * 3) / (self.alpha * 800 * np.pi * self.rho_c(self.cluster_z))) ** (1./3.)
    
    def r_2(self):

        return self.r200()/self.concentration

    def density(self, r):
        
        a =  gamma(3./self.a) * gammainc(3/self.a, (2/self.a) * self.concentration ** self.a)
        
        b = self.alpha * (400./3.) * self.rho_c(self.cluster_z)
        
        c = (self.concentration ** 3) * np.exp( - 2./self.a ) * (2/self.a) ** ((3 - self.a)/self.a)
        
        self.rho_2 = a ** (-1) * b * c
              
        x = r/self.r_2()
              
        return self.rho_2 * np.exp( - (2./self.a) * (x ** self.a - 1.) )
    