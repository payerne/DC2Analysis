import numpy as np
from astropy import units as u
from astropy import constants as const
import math

class Modeling():

    def __init__(self, M200, concentration, cluster_z, mass_def, cosmo):
    
        self.mass_def = mass_def
        self.M200 = M200
        self.concentration = concentration
        self.cosmo = cosmo
        self.cluster_z = cluster_z
        
        if self.mass_def == 'mean': self.alpha = self.cosmo.Om(self.cluster_z)
        else : self.alpha = 1.
        
   ############################
        
    def rho_c(self, z):

        return self.cosmo.critical_density(z).to(u.Msun / u.Mpc**3).value

    def delta_c(self, c):

        return (np.log(1 + c) - c/(1 + c))

    def r200(self):
        
        if self.mass_def == 'mean': Omega_m = self.cosmo.Om(self.cluster_z)
            
        else : Omega_m = 1

        return ((self.M200 * 3) / (self.alpha * 800 * np.pi * self.rho_c(self.cluster_z))) ** (1./3.)

    def rs(self, r200):

        return r200 / self.concentration
    
    def A(self,c):
        
        first = (c**3/self.delta_c(c))
        
        second = (200./3.)
        
        third = self.alpha * self.rho_c(self.cluster_z)
        
        return first * second * third
    
    def M(self,r):
        
        A, r200 = self.A(self.concentration), self.r200()
        
        r_s = self.rs(r200)
        
        x = r/r_s
        
        return A * 4 * np.pi * r_s ** 3 * self.delta_c(x) 
    
    def density(self,r):
        
        rho_3d = []
        
        A = self.A(self.concentration)
        
        r200 = self.r200()
        
        r_s = self.rs(r200)
        
        for R in r:
            
            rho_3d.append(A / ((R/r_s) * (1. + R/r_s) ** 2))
        
        return np.array(rho_3d)
    
    r"""
        shear
        
    """

    def kappa_u(self,x):

        def inf(x):

            first = 1/(x**2  - 1)
            second = 1 - (2./np.sqrt(1-x**2))*np.arctanh(np.sqrt((1-x)/(1 + x)))

            return first*second

        def equal(x):

            first = 1./3

            return first

        def sup(x):

            first = 1/(x**2 - 1)
            second = 1 - (2./np.sqrt(x**2-1))*np.arctan(np.sqrt((x - 1)/(x + 1)))

            return first*second


        if x > 1:

            return sup(x)

        if x == 1:

            return equal(x)

        if x < 1:

            return inf(x)

    
    def shear_u(self,x):
        
        ## three function for shear

        def ginf(x):

            racine = np.sqrt((1-x)/(1 + x))

            first = 8.*np.arctanh(racine)/(x**2*np.sqrt(1 - x**2))

            second = (4./x**2)*np.log(x/2)

            third = -2./(x**2 - 1)

            fourth = 4.*np.arctanh(racine)/((x**2 - 1)*np.sqrt(1 - x**2))

            return float(first + second + third + fourth)

        def gequal(x):

            first = 10./3 + 4*np.ln(1./2)

            return float(first)

        def gsup(x):

            racine = np.sqrt((x - 1)/(1 + x))

            first = 8.*np.arctan(racine)/(x**2*np.sqrt(x**2 - 1))

            second = (4./x**2)*np.log(x/2)

            third = -2./(x**2 - 1)

            fourth = 4.*np.arctan(racine)/((x**2 - 1)**(3./2))

            return float(first + second + third + fourth)

        
        if x > 1:

            return gsup(x)

        if x < 1:

            return ginf(x)

        if x == 1:

            return gequal(x)

    def critical_density(self, source_z): #Msun/Mpc


        first = self.cosmo.angular_diameter_distance(self.cluster_z).to(u.Mpc)

        second = self.cosmo.angular_diameter_distance(source_z).to(u.Mpc)

        third = self.cosmo.angular_diameter_distance_z1z2(self.cluster_z, source_z).to(u.Mpc)

        fourth = second/(first*third)

        G = const.G.to(u.Mpc**3 / (u.Msun * u.year**2))

        c = const.c.to(u.Mpc / u.year)

        fifth = c**2/(np.pi*4*G)

        return fifth*fourth
    
    def coeff(self, z_source):
        
        return self.rs(self.r200())*self.A(self.concentration)/self.critical_density(z_source).value
    
    def coeff_2(self):
        
        return self.rs(self.r200())*self.A(self.concentration)


    def shear(self, x, z_source): return self.coeff(z_source)*float(self.shear_u(x))


    def kappa(self, x, z_source): return (2) * float(self.kappa_u(x)) * self.coeff(z_source)
    
    
    def deltasigma(self, x): return self.coeff_2() * float(self.shear_u(x))


    def reduced_tangential_shear(self, r, source_z):

        r_200 = self.r200()

        r_s = self.rs(r_200)
        
        y = []
        
        for i, R in enumerate(r):

            x = R/r_s
            
            y.append(self.shear(x, source_z)/(1 - self.kappa(x, source_z)))
            
        return np.array(y)
    
    def excess_surface_density(self, r):
        
        r_200 = self.r200()

        r_s = self.rs(r_200)
        
        y = []
        
        for i, R in enumerate(r):

            x = R/r_s
            
            y.append(self.deltasigma(x))
            
        return np.array(y)            
    
    


        


