import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pca_resource import PCAResults
from models_class import Modelparam
from lr_model import getLR
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

    reg = getLR(selected_weights, Y_train, model).reshape(1,modelparam)
   
    Hedgeratio = np.matmul(reg,np.linalg.pinv(PC))
    
    
    return Hedgeratio



#     X=pd.DataFrame(X_train)
#     Y=pd.DataFrame(Y_train)
#     L = pd.concat([X, Y], axis=1)
#     cor = L.corr()
#     covar=np.cov(L,rowvar=False)
# #     print(covar)
#     u,s,vt=np.linalg.svd(covar)
#     eigen_vector = vt.T[:,:]
    
#     df_eigenv = pd.DataFrame(eigen_vector)
#     PC = pd.DataFrame(df_eigenv.iloc[:,0:modelparam])
#     EV = PC.iloc[-1,:]
#     EVs = PC.iloc[:modelparam,:]
#     HedgeRatio = np.matmul( EV,np.linalg.inv(EVs))
#     return HedgeRatio





