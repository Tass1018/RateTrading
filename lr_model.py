import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from scipy.optimize import minimize


def getLR(X, Y, model):
    
    reg=LinearRegression().fit(X,Y)
    coeff=np.squeeze(reg.coef_)
    sum_coeff=coeff.sum()

    norm_coeff = coeff/sum_coeff
    predic_y = np.matmul(X,coeff.T)
    error=np.squeeze(Y)-predic_y
    errorsq=np.square(error)
    error_sum=sum(errorsq)

    lassoParam, ridgeParam=model.lamb_index_index1, model.lamb_index_index2 
      
    l2_mape_model = CustomLinearModel(error_sum=error_sum,
        beta_init=coeff,
        X=X, Y=Y, regularization=[lassoParam,ridgeParam])
    
    l2_mape_model.fit()
    
    beta = l2_mape_model.beta

    return beta
    



class CustomLinearModel:
    """
    Linear model: Y = XB, fit by minimizing the provided loss_function
    with L2 regularization
    """
    def __init__(self, error_sum=None, 
                 X=None, Y=None, beta_init=None, 
                 regularization=[0,0]):
        self.regularization = regularization
        self.beta = None
        self.beta_init = beta_init
        self.X = X
        self.Y = Y
        self.error_sum=error_sum

    def l2_regularized_loss(self, beta):
        self.beta = beta
        return(self.error_sum + \
           self.regularization[0]*np.absolute(sum(np.array(self.beta))-1) + \
            sum(self.regularization[1]*np.array(self.beta)**2))       
    
    def fit(self, maxiter=250):
        
        # Initialize beta estimates (you may need to normalize
        # your data and choose smarter initialization values
        # depending on the shape of your loss function)
        if type(self.beta_init)==type(None):
            # set beta_init = 1 for every feature
            self.beta_init = np.array([1]*self.X.shape[1])
        else: 
            # Use provided initial values
            pass

        if self.beta!=None and all(self.beta_init == self.beta):
            print("Model already fit once; continuing fit with more itrations.")
        res = minimize(self.l2_regularized_loss, self.beta_init)
        self.beta = res.x
        self.beta_init = self.beta