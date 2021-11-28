import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import eigh

def second_derivative_matrix(theta, model):
    
    delta = 1e-5
    first_derivative = []
    second_derivative = np.zeros([len(theta),len(theta)])
    first_derivative = np.zeros(len(theta))
    
    def derive_i(theta, i):
                delta_i = np.zeros(len(theta))
                delta_i[i] = delta
                res = (model(np.array(theta) + delta_i/2) - model(np.array(theta) - delta_i/2))/delta
                return res
    for i in range(len(theta)):
        first_derivative[i] = derive_i(theta, i)
        for j in range(len(theta)):
            if i >= j:
                delta_j = np.zeros(len(theta))
                delta_j[j] = delta
                deriv_ij = (derive_i(np.array(theta) + delta_j/2, i) - derive_i(np.array(theta) - delta_j/2, i))/delta
                second_derivative[i,j] = deriv_ij
                second_derivative[j,i] = second_derivative[i,j]
    return second_derivative
            
def Fisher_Matrix(theta, model, cov, delta = 1e-5):
    r"""
    Attributes:
    -----------
    theta : array
        the values of model parameters used to evaluate the Fisher matrix
    model : fct
        the model taht depends on the parameters thta
    cov : array
        the covariance matrix of the observed data
        
    Returns:
    --------
    Fisher_matrix_inv : array
        the covariance matrix of parameters theta
   """
    zeros = np.zeros(len(theta))
    derivative = []
    for i in range(len(theta)):
        delta_i = np.zeros(len(theta))
        delta_i[i] = delta
        derivative.append((model(np.array(theta) + delta_i/2) \
                           - model(np.array(theta) - delta_i/2))/delta)
    Fisher_matrix = np.zeros([len(theta), len(theta)])
    for i in range(len(theta)):
        for j in range(len(theta)):
            Fisher_matrix[i,j] = np.sum(derivative[i] * np.linalg.inv(cov).dot(derivative[j]))
    return Fisher_matrix

def S_Fisher_Matrix(theta, model, cov):
    r"""
    https://arxiv.org/pdf/1606.06455.pdf
    """
    zeros = np.zeros(len(theta))
    delta = 1e-2
    first_derivative = []
    second_derivative = np.zeros([len(theta),len(theta), len(cov[:,0])])
    first_derivative = np.zeros([len(theta), len(cov[:,0])])
    def derive_i(theta, i):
                delta_i = np.zeros(len(theta))   
                delta_i[i] = delta
                res = (model(np.array(theta) + delta_i/2) - \
                       model(np.array(theta) - delta_i/2))/delta
                return res      
    for i in range(len(theta)):
        first_derivative[i] = derive_i(theta, i) 
        for j in range(len(theta)):
            delta_j = np.zeros(len(theta))
            delta_j[j] = delta
            deriv_ij = (derive_i(np.array(theta) + delta_j/2, i) - \
                        derive_i(np.array(theta) - delta_j/2, i))/delta
            second_derivative[i,j,:] = deriv_ij
    S_matrix = np.zeros([len(theta),len(theta),len(theta)])
    for k in range(len(theta)):
        for l in range(len(theta)): 
            for m in range(len(theta)):
                S_matrix[k,l,m] = np.sum(second_derivative[k,l]* \
                                         np.linalg.inv(cov).dot(first_derivative[m]))
    return S_matrix

def Q_Fisher_Matrix(theta, model, cov):
    r"""
    https://arxiv.org/pdf/1606.06455.pdf
    """
    zeros = np.zeros(len(theta))
    delta = 1e-7
    second_derivative = np.zeros([len(theta),len(theta), len(cov[:,0])])
    def derive_i(theta, i):
                delta_i = np.zeros(len(theta))
                delta_i[i] = delta
                res = (model(np.array(theta) + delta_i/2) - model(np.array(theta) - delta_i/2))/delta
                return res
    for i in range(len(theta)):
        for j in range(len(theta)):
            delta_j = np.zeros(len(theta))
            delta_j[j] = delta
            deriv_ij = (derive_i(np.array(theta) + delta_j/2, i) - derive_i(np.array(theta) - delta_j/2, i))/delta
            second_derivative[i,j,:] = deriv_ij
    Q_matrix = np.zeros([len(theta),len(theta),len(theta), len(theta)])
    for k in range(len(theta)):
        for l in range(len(theta)): 
            for m in range(len(theta)):
                for n in range(len(theta)):
                    Q_matrix[k,l,m,n] = np.sum(second_derivative[k,l]*np.linalg.inv(cov).dot(second_derivative[m,n]))
    return Q_matrix
    return lnL_Gaussian + lnL_Q_Gaussian + lnL_S_Gaussian

def lnL_Fisher(theta, MLE, Fisher_matrix):
    dtheta = theta - MLE
    lnL_Gaussian = -0.5*np.sum(dtheta*Fisher_matrix.dot(dtheta))
    return lnL_Gaussian 

def lnL_S_Fisher(theta, MLE, S_matrix):
    dtheta = theta - MLE
    S = 0
    for i in range(len(theta)):
        for j in range(len(theta)): 
            for k in range(len(theta)):
                S += S_matrix[i,j,k]*dtheta[i]*dtheta[j]*dtheta[k]
    lnL_S_Gaussian = -(1./2.)*S

    return lnL_S_Gaussian

def lnL_Q_Fisher(theta, MLE, Q_matrix):
    dtheta = theta - MLE
    Q = 0
    for i in range(len(theta)):
        for j in range(len(theta)): 
            for k in range(len(theta)):
                for l in range(len(theta)):
                        Q += Q_matrix[i,j,k,l]*dtheta[i]*dtheta[j]*dtheta[k]*dtheta[l]
    lnL_Q_Gaussian = -(1./8.)*Q
    
    return lnL_Q_Gaussian

def decorrelation_rotation_matrix_bivariate(cov):
    def f_optimize(p):
        a,b = p
        R = np.array([[a,-b],[b,a]])
        detR = np.linalg.det(R)
        R_inv = np.linalg.inv(R)
        Lambda = np.dot(R_inv, np.dot(cov_inv, R))
        res = Lambda - diag
    return res[0,0], detR - 1.
    res = fsolve(f_optimize, np.random.randn(2))
    Rotation_matrix = np.array([[res[0], -res[1]],[res[1], res[0]]])
    return Rotation_matrix
    