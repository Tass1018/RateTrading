import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize

from openpyxl import load_workbook
wb = load_workbook(filename = 'DataDownload.xlsx')

"""
    getPCA: Dimension reduction on the hedge points and 
            do a linear regression with target pt data to get hedge ratios

"""
def getPCA(X_train, Y_train, model):
    modelparam = model.number_of_components_index
    X_train=pd.DataFrame(X_train)
    X_ctd=X_train-np.mean(X_train,axis=0)
    X_covar=np.cov(X_ctd,rowvar=False)
    u,s,vt=np.linalg.svd(X_covar)
    e_vectors=vt.T[:,:]
    df_eigenv=pd.DataFrame(e_vectors)
    PC = pd.DataFrame(df_eigenv.iloc[:,0:modelparam])

    weights=np.squeeze(np.matmul(X_train,np.linalg.inv(df_eigenv.T)))
    selected_weights = pd.DataFrame(weights.iloc[:,0:modelparam])
   
    df_e_vectors=pd.DataFrame(selected_weights)

    reg = simpleLR(selected_weights, Y_train).reshape(1,modelparam)
   
    Hedgeratio = np.matmul(reg,np.linalg.pinv(PC))
    
    HR_optimized = PCA_OP(Hedgeratio, X_train, Y_train, model)
    
    
    return HR_optimized



def simpleLR(X,Y):
    reg=LinearRegression().fit(X,Y)
    coeff=np.squeeze(reg.coef_)
    return coeff

"""
    PCA_OP(Hedgeratio, X, Y, model)
    Hedgeratio: initial beta values produced by simple linear regression
    X, Y: training data for regression
    model: model used 

"""
def PCA_OP(Hedgeratio, X, Y, model):
    coeff=Hedgeratio
    
    predic_y = np.matmul(X,coeff.T)
    
    error=Y-predic_y

    errorsq=np.square(error)

    error_sum=sum(errorsq)
    
    lassoParam, ridgeParam=model.lamb_index_index1, model.lamb_index_index2 
      
    l2_mape_model = CustomLinearModel(error_sum=error_sum,
        beta_init=coeff,
        X=X, Y=Y, regularization=[lassoParam,ridgeParam])
    
    l2_mape_model.fit()
    
    beta = l2_mape_model.beta

    return beta   
    
###################################################################################    
    
    
    
from scipy.optimize import minimize

"""
CustomLinearModel : It is a class that can costume the cost function to optimize 
                    the beta values from simple linear regression.


    __init__(self): regularization is a list contains lasso lambda and ridge lambda values.
                    beta_init is the initial beta values produced by simple linear regression.
                    error_sum is the result of OSL function.

    Update(self,modeldic): update all values before running the model
"""
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

    """
    L2 regularization
    ans: value of cost function
    cost function = OSL + lasso function + ridge function
    """
    def l2_regularized_loss(self, beta):
        self.beta = beta
        ans=(self.error_sum + \
           self.regularization[0]*np.absolute(sum(np.array(self.beta))-1) + \
            sum(self.regularization[1]*np.array(self.beta)**2))
        
        return ans
            
        
            
    """
    fit: find the minimum cost function betas
    ans: value of cost function
    cost function = OSL + lasso function + ridge function
    """
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


    
    
    
    