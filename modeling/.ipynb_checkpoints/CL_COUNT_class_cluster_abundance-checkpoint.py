import numpy as np
import pyccl as ccl
import numpy as np
import scipy
from scipy import stats
from scipy.integrate import quad,simps, dblquad
from scipy import interpolate

class ClusterAbundance():
    r"""
        1. computation of the cosmological prediction for cluster abundance cosmology, for 
            a. cluster count in mass and redhsift intervals (binned approach)
            b. cluster count with individual masses and redshifts (un-binned approach)
            c. cluster count in mass proxy and redhsift intervals (binned approach)
            d. cluster count with individual mass proxies and redshifts (un-binned approach)
        Uses Core Cosmology Library (arXiv:1812.05995) as backend.
    """
    def ___init___(self):
        self.name = 'Likelihood for cluster count Cosmology'
        
    def set_cosmology(self, cosmo = 1, massdef = None, hmd = None):
        r"""
        Attributes:
        ----------
        cosmo : CCL cosmology object
        mass_def: CCL object
            mass definition object of CCL
        hmd: CCL object
            halo mass distribution object from CCL
        """
        self.cosmo = cosmo
        self.massdef = massdef
        self.hmd = hmd
        #self.massdef = ccl.halos.massdef.MassDef('vir', 'critical', c_m_relation=None)
        #self.hmd = ccl.halos.hmfunc.MassFuncDespali16(self.cosmo, mass_def=self.massdef)
        
    def dndlog10M(self, log10M, z):
        r"""
        Attributes:
        -----------
        log10M : array
            \log_{10}(M), M dark matter halo mass
        z : float
            halo redshift
        Returns:
        --------
        hmf : array
            halo mass function for the corresponding masses and redshift
        """
        hmf = self.hmd.get_mass_function(self.cosmo, 10**np.array(log10M), 1./(1. + z))
        return hmf

    def dVdzdOmega(self,z):
        r"""
        Attributes:
        ----------
        z : float
            redshift
        Returns:
        -------
        dVdzdOmega_value : float
            differential comoving volume 
        """
        a = 1./(1. + z)
        da = ccl.background.angular_diameter_distance(self.cosmo, a)
        ez = ccl.background.h_over_h0(self.cosmo, a) 
        dh = ccl.physical_constants.CLIGHT_HMPC / self.cosmo['h']
        dVdzdOmega_value = dh * da * da/( ez * a ** 2)
        return dVdzdOmega_value

    def compute_multiplicity_grid_MZ(self, z_grid = 1, logm_grid = 1):
        r"""
        Attributes:
        -----------
        z_grid : array
            redshift grid
        logm_grid : array
            logm grid
        Returns:
        --------
        dN_dzdlogMdOmega : array
            tabulated multiplicity function over the redshift and logmass grid
        dzdlogMdOmega_interpolation : function
            interpolated function over the tabulated multiplicity grid
        """
        self.z_grid = z_grid
        self.logm_grid = logm_grid
        grid = np.zeros([len(self.logm_grid), len(self.z_grid)])
        for i, z in enumerate(self.z_grid):
            grid[:,i] = self.dndlog10M(self.logm_grid ,z) * self.dVdzdOmega(z)
        self.dN_dzdlogMdOmega = grid
        self.dNdzdlogMdOmega_interpolation = interpolate.interp2d(self.z_grid, 
                                                                self.logm_grid, 
                                                                self.dN_dzdlogMdOmega, 
                                                                kind='cubic')
        
    def Cluster_Abundance_MZ(self, Redshift_bin = [], Proxy_bin = [], method = 'dblquad_interp'): 
        r"""
        returns the predicted number count in mass-redshift bins
        Attributes:
        -----------
        Redshift_bin : list of lists
            list of redshift bins
        Proxy_bin : list of lists
            list of mass bins
        method : str
            method to be used for the cluster abundance prediction
            "simps": use simpson integral of the tabulated multiplicity
            "dblquad_interp": integer interpolated multiplicity function
            "exact_CCL": use scipy.dblquad to integer CCL multiplicity function
        Returns:
        --------
        N_th_matrix: ndarray
            matrix for the cluster abundance prediction in redshift and mass bins
        """
        N_th_matrix = np.zeros([len(Redshift_bin), len(Proxy_bin)])
        if method == 'dblquad_interp':
            for i, proxy_bin in enumerate(Proxy_bin):
                for j, z_bin in enumerate(Redshift_bin):
                    N_th_matrix[j,i] = self.sky_area * dblquad(self.dNdzdlogMdOmega_interpolation, 
                                                   proxy_bin[0], proxy_bin[1], 
                                                   lambda x: z_bin[0], 
                                                   lambda x: z_bin[1])[0]
                    
        if method == 'simps':
            index_proxy = np.arange(len(self.logm_grid))
            index_z = np.arange(len(self.z_grid))
            for i, proxy_bin in enumerate(Proxy_bin):
                mask_proxy = (self.logm_grid >= proxy_bin[0])*(self.logm_grid <= proxy_bin[1])
                proxy_cut = self.logm_grid[mask_proxy]
                index_proxy_cut = index_proxy[mask_proxy]
                proxy_cut[0], proxy_cut[-1] = proxy_bin[0], proxy_bin[1]
                for j, z_bin in enumerate(Redshift_bin):
                    z_down, z_up = z_bin[0], z_bin[1]
                    mask_z = (self.z_grid >= z_bin[0])*(self.z_grid <= z_bin[1])
                    z_cut = self.z_grid[mask_z]
                    index_z_cut = index_z[mask_z]
                    z_cut[0], z_cut[-1] = z_down, z_up
                    integrand = np.array([self.dN_dzdlogMdOmega[:,k][mask_proxy] for k in index_z_cut])
                    N_th = self.sky_area * simps(simps(integrand, proxy_cut), z_cut)
                    N_th_matrix[j,i] = N_th
                    
        if method == 'exact_CCL':
            def dN_dzdlogMdOmega(logm, z):
                return self.sky_area * self.dVdzdOmega(z) * self.dndlog10M(logm, z)
            for i, proxy_bin in enumerate(Proxy_bin):
                for j, z_bin in enumerate(Redshift_bin):
                    N_th_matrix[j,i] = scipy.integrate.dblquad(dN_dzdlogMdOmega, 
                                                               z_bin[0], z_bin[1], 
                                                               lambda x: proxy_bin[0], 
                                                               lambda x: proxy_bin[1])[0]

        return N_th_matrix
    
    def multiplicity_function_individual_MZ(self, z = .1, logm = 14, method = 'interp'):
        r"""
        Attributes:
        -----------
        z: array
            list of redshifs
        logm: array
            list of dark matter halo masses
        method: str
            method to use to compute multiplicity function
            "interp": use interpolated multiplicity function
            "exact_CCL": idividual CCL prediction
        Returns:
        --------
        dN_dzdlogMdOmega : array
            multiplicity function for the corresponding redshifts and masses
        """
        if method == 'interp':
            dN_dzdlogMdOmega_fct = interpolate.RectBivariateSpline(self.logm_grid, self.z_grid, 
                                                                   self.dN_dzdlogMdOmega)
            dN_dzdlogMdOmega = dN_dzdlogMdOmega_fct(logm, z, grid = False)    
        if method == 'exact_CCL':
            dN_dzdlogMdOmega = np.zeros(len(z))
            for i, z_ind, logm_ind in zip(np.arange(len(z)), z, logm):
                dN_dzdlogMdOmega[i] = self.dndlog10M(logm_ind, z_ind) * self.dVdzdOmega(z_ind)
        return dN_dzdlogMdOmega

    def compute_cumulative_grid_ProxyZ(self, Proxy_bin = [], proxy_model = 1, 
                                       z_grid = 1, logm_grid = 1):
        r"""
        Attributes:
        -----------
        lnLambda_bin: list of 2D arrays
            list of proxy bins
        proxy_model: object
            object from class with mass-proxy relation methods
        Returns:
        --------
        cumulative_proxy_grid : array
            2D array of the proxy cumulative in proxy bins on a mass-redshift grid
        cumulative_proxy_grid_interp : array
            interpolation of the proxy cumulative in proxy bins
        """
        cumulative_grid = []
        cumulative_grid_interp = []
        for proxy_bin in Proxy_bin:
            cdf = np.zeros([len(logm_grid), len(z_grid)])
            for i, z_value in enumerate(z_grid):
                cdf[:,i] = proxy_model.integral_in_bin(proxy_bin, z_value, logm_grid)
            cdf_interp = interpolate.interp2d(z_grid, 
                                            logm_grid, 
                                            cdf, 
                                            kind='cubic')
            cumulative_grid.append(cdf)
            cumulative_grid_interp.append(cdf_interp)
        self.cumulative_proxy_grid = np.array(cumulative_grid)
        self.cumulative_proxy_grid_interp = np.array(cumulative_grid_interp)
        
    def compute_pdf_grid_ProxyZ(self, proxy, z, proxy_model = 1,
                               z_grid = 1, logm_grid = 1):
        r"""
        Attributes:
        -----------
        z: array
            list of redshifs
        proxy: array
            list of dark matter halo masses
        proxy_model: object
            object from class with mass-proxy relation methods
        Returns:
        --------       
        pdf : array
            Density function table as a function of mass 
            for individual redshifts and proxy
        """
        pdf = np.zeros([len(z), len(logm_grid)])
        for i, zs, proxys in zip(np.arange(len(z)), z, proxy):
            pdf[i,:] = proxy_model.P(proxys, zs, logm_grid)
        self.pdf = pdf

    def Cluster_Abundance_ProxyZ(self, Redshift_bin = [], Proxy_bin = [], logm_limit = [], 
                                 proxy_model = None, method = 'dblquad_interp'): 
        r"""
        Attributes:
        -----------
        Redshift_bin: list of lists
            list of redshift bins
        Proxy_bin: list of lists
            list of mass-proxy bins
        method: str
            method to be used for the cluster abundance prediction
        proxy_model: object
            object from class with mass-proxy relation methods
        Returns:
        --------
        N_th_matrix: ndarray
            Cluster abundance prediction in redshift and proxy bins
        """     
        N_th_matrix = np.zeros([len(Redshift_bin), len(Proxy_bin)])
        if method == 'simps':
            index_z = np.arange(len(self.z_grid))
            index_logm = np.arange(len(self.logm_grid))
            mask_logm = (self.logm_grid >= logm_limit[0])*(self.logm_grid <= logm_limit[1])
            logm_cut = self.logm_grid[mask_logm]
            index_mask_logm_cut = index_logm[mask_logm]
            logm_cut[0], logm_cut[-1] = logm_limit[0], logm_limit[1]
            for j, z_bin in enumerate(Redshift_bin):
                z_down, z_up = z_bin[0], z_bin[1]
                mask_z = (self.z_grid >= z_bin[0])*(self.z_grid <= z_bin[1])
                z_cut = self.z_grid[mask_z]
                index_z_cut = index_z[mask_z]
                z_cut[0], z_cut[-1] = z_down, z_up
                integrand_mass = np.array([self.dN_dzdlogMdOmega[:,k][mask_logm] for k in index_z_cut])
                for i, proxy_bin in enumerate(Proxy_bin):
                    integrand_cumulative = np.array([self.cumulative_proxy_grid[i][:,k][mask_logm] for k in index_z_cut])
                    N_th = self.sky_area * simps(simps(integrand_mass * integrand_cumulative, 
                                                       logm_cut),
                                                 z_cut)
                    N_th_matrix[j,i] = N_th
                    
        if method == 'dblquad_interp':
            for i, proxy_bin in enumerate(Proxy_bin):
                for j, z_bin in enumerate(Redshift_bin):
                    def fct_to_integer(z, logm):
                        res1 = self.cumulative_proxy_grid_interp[i](z, logm)
                        res2 = self.dNdzdlogMdOmega_interpolation(z, logm)
                        return res1 * res2
                    N_th_matrix[j,i] = self.sky_area * dblquad(fct_to_integer, 
                                                   logm_limit[0], logm_limit[1], 
                                                   lambda x: z_bin[0], lambda x: z_bin[1],
                                                               epsabs=1.49e-07)[0]
                    
        if method == 'exact_CCL':
            for i, proxy_bin in enumerate(Proxy_bin):
                for j, z_bin in enumerate(Redshift_bin):
                    def dN_dzdproxydOmega(logm, z):
                        res1 = self.sky_area * self.dVdzdOmega(z) * self.dndlog10M(logm, z)
                        res2 = proxy_model.integral_in_bin(proxy_bin, z, logm)
                        return res1 * res2
                    N_th_matrix[j,i] = scipy.integrate.dblquad(dN_dzdproxydOmega, 
                                                               z_bin[0], z_bin[1], 
                                                               lambda x: logm_limit[0], 
                                                               lambda x: logm_limit[1])[0]
        return N_th_matrix

    def multiplicity_function_individual_ProxyZ(self, z = .1, proxy = 1, 
                                                proxy_model = 1, method = 'simps'):
        r"""
        Attributes:
        -----------
        z: array
            list of redshifs
        proxy: array
            list of dark matter halo masses
        proxy_model: object
            object from proxy class
        method: str
            method to use to compute multiplicity function
            "simps": use simpson integral
            "exact_CCL": use scipy.dblquad 
        Returns:
        --------       
        dN_dzdlogMdOmega : array
            multiplicity function for the corresponding redshifts and masses
        """
        if method == 'simps':
            z_sort = np.sort(z)
            index_sort = np.argsort(z)
            index_z = np.arange(len(z))
            multiplicity_logm = self.dNdzdlogMdOmega_interpolation(z_sort, self.logm_grid).T
            dndproxy = scipy.integrate.simps(multiplicity_logm * self.pdf[np.argsort(z)], x = self.logm_grid, axis = 1)
            dndproxy = dndproxy[np.argsort(index_sort)]
            return dndproxy
        
        if method == 'exact_CCL':
            dndproxy = []
            for zs, proxys in zip(z, proxy):
                def __integrand__(logm):
                    res1 = proxy_model.P(proxys, zs, logm)
                    res2 = self.dndlog10M(logm, zs)*self.dVdzdOmega(zs)
                    return res1*res2
                dndproxy.append(scipy.integrate.quad(__integrand__, self.logm_grid[0], self.logm_grid[-1])[0])
            return np.array(dndproxy)
        