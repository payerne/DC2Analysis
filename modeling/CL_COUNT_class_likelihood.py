import numpy as np
import numpy as np
import healpy
import scipy
from scipy import stats
from scipy.special import erfc
from scipy.stats import poisson
from scipy.stats import multivariate_normal
from scipy.integrate import quad,simps, dblquad

def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]

class Likelihood():
    r"""
        compute likelihood :
            a. for the binned gaussian case
            b. for the binned poissonian case
            c. for the un-binned poissonian case
    """
    def ___init___(self):
        self.name = 'Likelihood for cluster count Cosmology'
        
        
    def Gaussian(self, x, mu, var_SSC):
        r"""
        Attributes:
        -----------
        x: array
            variable along the x axis
        mu: float
            mean of the Gaussian distribution
        var_SSC: float
            variance of the Gaussian distrubution
        Returns:
        --------
        g: array
            Gausian probability density function
        """
        return np.exp(-.5*(x-mu)**2/var_SSC)/np.sqrt(2*np.pi*var_SSC)
        
    def poissonian(self, n, mu):
        r"""
        Attributes:
        -----------
        n: array
            variable along the n axis
        mu: float
            mean of the Poisson distribution
        Returns:
        --------
        p: array
            Poisson probability function
        """
        rv = poisson(mu)
        return rv.pmf(n)
        
    def lnLikelihood_Binned_Poissonian(self, N_th_matrix, N_obs_matrix):
        r"""
        returns the value of the log-likelihood for Poissonian binned approach
        Attributes:
        -----------
        N_th_matrix: array
            cosmological prediction for binned cluster abundance
        N_obs_matrix:
            observed binned cluster abundance
        Returns:
        --------
        add attributes with total log-likelihood for Poissonian binned approach
        """
        lnL_Poissonian = np.sum(N_obs_matrix.flatten() * np.log(N_th_matrix.flatten()) - N_th_matrix.flatten())
        self.lnL_Binned_Poissonian = lnL_Poissonian

    def lnLikelihood_Binned_Gaussian(self, N_th_matrix, N_obs_matrix, covariance_matrix):
        r"""
        Attributes:
        -----------
        N_th_matrix: array
            cosmological prediction for binned cluster abundance
        N_obs_matrix:
            observed binned cluster abundance
        covariance_matrix: array
            full covariance matrix for binned cluster abundance
        Returns:
        --------
        add attributes with total log-likelihood for Gaussian binned approach
        """
        delta = (N_obs_matrix - N_th_matrix).flatten()
        inv_covariance_matrix = np.linalg.inv((covariance_matrix))
        lnL_Gaussian = -0.5*np.sum(delta*inv_covariance_matrix.dot(delta)) 
        self.lnL_Binned_Gaussian = lnL_Gaussian
        
    def lnLikelihood_Binned_MPG(self, N_th_matrix, N_obs_matrix, sample_variance):
        r"""
        Attributes:
        -----------
        N_th_matrix: array
            cosmological prediction for binned cluster abundance
        N_obs_matrix:
            observed binned cluster abundance
        covariance_matrix: array
            sample variance for binned cluster abundance
        Returns:
        --------
        add attributes with total log-likelihood for Gaussian & Poissonian mixture binned approach
        """
        def GP_product(x, n, mu, var_SSC):
            r"""
            Attributes:
            -----------
            x: array
                variable along the integrand axis
            n: int
                observed cluster count
            mu: float
                cosmological prediction for cluster count
            var_SSC: float
                variance of the gaussian
            Returns:
            --------
            integrand: array
            """
            return self.poissonian(n, x) * self.Gaussian(x, mu, var_SSC)
    
        def P_MPG_delta(N_obs, mu, var_SSC):
            r"""
            Attributes:
            -----------
            N_array: array
                cluster count axis (int values)
            mu: float
                cluster abuncance cosmological prediction
            var_SSC: folat
                SSC variance of cluster count
            Returns:
            --------
            p_mvp: array
                Gaussian/Poisson mixture probability along the overdensity axis
            r"""
            u_array = np.linspace(0, 1, 500)
            n_sigma_delta = 3
            K1 = (1./np.sqrt(2.*np.pi*var_SSC))
            K2 = np.sqrt(1./var_SSC)
            K3 = np.sqrt(np.pi/2.)
            K4 = mu*K2/np.sqrt(2.)
            K = 1 - (K3*erfc(K4)/K2)*K1
            up = mu + n_sigma_delta*np.sqrt(var_SSC)
            down = mu - n_sigma_delta*np.sqrt(var_SSC)
            u_axis = u_array*(up - down) + down
            p_mvp = np.zeros( len(u_array) )
            res = GP_product(u_axis, N_obs, mu, var_SSC)
            res = np.where(u_axis >= 0, res, 0)
            return res.T/K
    
        _integrand_ = np.zeros([len(N_obs_matrix), len(u_array)])
        L = 1
        var_list = sample_variance
        for i, nth in enumerate(N_th_matrix.flatten()):
            p = P_MPG_delta(N_obs_matrix[i], nth, var_list[i])
            #change of variable
            alpha = 2*n_sigma_delta*np.sqrt(var_list[i])/L
            _integrand_[i,:] = alpha * p
        res = simps(_integrand_, u_array)
        self.lnL_Binned_MPG = np.sum(np.log(res))
        
    def lnLikelihood_UnBinned_Poissonian(self, dN_dzdlogMdOmega, N_tot):
        r"""
        Attributes:
        -----------
       dN_dzdlogMdOmega: array
            cosmological prediction for multiplicu-ity function
        N_tot: float
            cosmological prediction for total number of cluster
        Returns:
        --------
        add attributes with total log-likelihood for Poissonian unbinned approach
        """
        self.lnL_UnBinned_Poissonian = np.sum(np.log(dN_dzdlogMdOmega)) - N_tot