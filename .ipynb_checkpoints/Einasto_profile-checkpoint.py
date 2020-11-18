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
        
        self.rho_2 = a **(-1) * b * c
              
        x = r/self.r_2()
              
        return self.rho_2 * np.exp( - (2./self.a) * (x ** self.a - 1.) )
    
    r"""
    
        
        cosmoCCL = ccl.Cosmology(Omega_c=0.265 - 0.0448, Omega_b=0.0448,
                      h=0.71, sigma8 = 0.8, n_s = 1)
        
                 
        if self.mass_def == 'mean': self.alpha = self.cosmo.Om(self.cluster_z)
            
        else : self.alpha = 1.
        
        def rho_c(z):

            return self.cosmo.critical_density(z).to(u.Msun / u.Mpc**3).value
    
        def r200():

            return ((self.M200 * 3) / (self.alpha * 800 * np.pi * self.rho_c(self.cluster_z))) ** (1./3.)
    
        def r_2():
        
            return self.r200()/self.concentration
        
        x_ = cosmo.Ode(self.cluster_z)
            
        Dc = 18*np.pi**2 - 82*x_ - 39*x_**2
        
        self.Dc = Dc
          
        def f(M):
            
            nu = 1.686/ccl.power.sigmaM(cosmoCCL, M, 1/(1 + self.cluster_z))
            
            a = 0.0095*nu**2 + 0.155
            
            r_vir = ( (M * 3) / (rho_c(self.cluster_z) * 4 * np.pi * Dc) ) ** (1./2.)
            
            x = r_vir/r_2()
            
            a1 =   gamma(3./a) * gammainc (3./a, (2./a) * self.concentration ** a)
        
            a2 =   gamma(3./a) * gammainc (3./a, (2./a) * (x) ** a)
            
            Mr_vir = self.M200 * a2/a1
            
            return M - Mr_vir
        
        M_vir = fsolve(func = f, x0 = self.M200)[0]
        
        self.M_vir = M_vir
        
        nu = 1.686/ccl.power.sigmaM(cosmoCCL, M_vir, 1/(1 + self.cluster_z)) 
            
        self.a = 0.0095*nu**2 + 0.155
    
    """