import numpy as np


def Fisher_Matrix(theta, model,cov):
    
    r"""
    https://arxiv.org/pdf/1606.06455.pdf
    """
    
    zeros = np.zeros(len(theta))
    
    delta = 1e-7
    
    derivative = []
    
    for i in range(len(theta)):
        
        delta_i = np.zeros(len(theta))
        
        delta_i[i] = delta
        
        derivative.append((model(np.array(theta) + delta_i/2) - model(np.array(theta) - delta_i/2))/delta)
        
    Fisher_matrix = np.zeros([len(theta), len(theta)])
    
    for i in range(len(theta)):
        
        for j in range(len(theta)):
            
            Fisher_matrix[i,j] = np.sum(derivative[i]*np.linalg.inv(cov).dot(derivative[j]))
            
    Fisher_matrix_inv = np.linalg.inv(Fisher_matrix)
    
    return Fisher_matrix_inv

def S_Fisher_Matrix(theta, model, cov):
    
    r"""
    https://arxiv.org/pdf/1606.06455.pdf
    """
    
    zeros = np.zeros(len(theta))
    
    delta = 1e-7
    
    first_derivative = []
    
    second_derivative = np.zeros([len(theta),len(theta), len(cov[:,0])])
    
    first_derivative = np.zeros([len(theta), len(cov[:,0])])
    
    def derive_i(theta, i):
        
                delta_i = np.zeros(len(theta))
            
                delta_i[i] = delta
                
                res = (model(np.array(theta) + delta_i/2) - model(np.array(theta) - delta_i/2))/delta
                
                return res
            
    for i in range(len(theta)):
            
        first_derivative[i] = derive_i(theta, i)
        
        for j in range(len(theta)):
        
            delta_j = np.zeros(len(theta))
            
            delta_j[j] = delta
            
            deriv_ij = (derive_i(np.array(theta) + delta_j/2, i) - derive_i(np.array(theta) - delta_j/2, i))/delta
            
            second_derivative[i,j,:] = deriv_ij
            
    S_matrix = np.zeros([len(theta),len(theta),len(theta)])
    
    for k in range(len(theta)):
        
        for l in range(len(theta)): 
            
            for m in range(len(theta)):
                
                S_matrix[k,l,m] = np.sum(second_derivative[k,l]*np.linalg.inv(cov).dot(first_derivative[m]))
                
    return np.linalg.inv(S_matrix)