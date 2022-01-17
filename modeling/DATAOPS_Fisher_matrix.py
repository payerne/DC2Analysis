import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import eigh

class Forecast():
    
    def ___init___(self):
        self.name = 'Forecast'
        
    def first_derivative(self, theta, model, shape_model, delta = 1e-5):
        r"""
        ref : https://en.wikipedia.org/wiki/Finite_difference
        Attributes:
        -----------
        theta : array
            parameter values to evaluate first dreivative
        model : fct
            model to compute second derivative
        Returns:
        --------
        sec : array
            array of first derivative
        """
        first = np.zeros([len(theta)] + list(shape_model))
        for i in range(len(theta)):
            delta_i = np.zeros(len(theta))
            delta_i[i] = delta
            first[i] = (model(theta + delta_i/2) - model(theta - delta_i/2))/delta
        return first
    
    def second_derivative(self, theta, model, shape_model, delta = 1e-5):
        r"""
        ref : https://en.wikipedia.org/wiki/Finite_difference
        Attributes:
        -----------
        theta : array
            parameter values to evaluate second dreivative
        model : fct
            model to compute second derivative
        Returns:
        --------
        sec : array
            array of second derivative
        """
        sec = np.zeros([len(theta),len(theta)] + list(shape_model))
        for i in range(len(theta)):
            delta_i = np.zeros(len(theta))
            delta_i[i] = delta
            for j in range(len(theta)):
                delta_j = np.zeros(len(theta))
                delta_j[j] = delta
                if i == j:
                    sec[i,j] = (model(theta+delta_i) - \
                                2*model(theta) + \
                                model(theta-delta_i))/delta**2
                elif i > j:
                    sec[i,j] = (model(theta+delta_i+delta_j) - \
                                model(theta+delta_i-delta_j) - \
                                model(theta-delta_i+delta_j) + \
                                model(theta-delta_i-delta_j))/(4*delta**2)
                    sec[j,i] = sec[i,j]
        return sec
    
    def Fisher_Matrix_Gaussian(self, theta, model, cov, delta = 1e-5):
        r"""
        Attributes:
        -----------
        theta : array
            the values of model parameters used to evaluate the Fisher matrix
        model : fct
            the model
        cov : array
            the covariance matrix of the observed data
        Returns:
        --------
        Fisher_matrix : array
            the Fisher matrix for the model parameters
       """
        Fisher_matrix = np.zeros([len(theta), len(theta)])
        shape_model = cov.diagonal().shape
        dd = self.first_derivative(theta, model, shape_model, delta = 1e-5)
        for i in range(len(theta)):
            for j in range(len(theta)):
                Fisher_matrix[i,j] = np.sum(dd[i] * np.linalg.inv(cov).dot(dd[j]))
        return Fisher_matrix

    r"""
    def second_derivative_matrix_(self, theta, model, delta = 1e-5):
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
    
    def second_derivative_matrix_new_(self, theta, model, delta = 1e-5):
        model_true = model(theta)
        
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
    

    def S_Fisher_Matrix(theta, model, cov):
       
        #https://arxiv.org/pdf/1606.06455.pdf
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
     
       ## https://arxiv.org/pdf/1606.06455.pdf
     
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
    """