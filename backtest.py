import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from lr_model import getLR
from pca_model import getPCA
from models_class import DataParam, Modelparam
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
import plotly.express as px
import plotly.graph_objects as go

def MidBackTest(curr,hedge_box):
    raw_data = pd.read_excel('//SVRSG001RPS01.asia.corp.anz.com/huy11$/My Documents/project/DataDownload.xlsm',sheet_name=curr,index_col='Dates')
    
    raw_data=raw_data.iloc[:,0:19]
    raw_data=raw_data.dropna()
    X=raw_data.copy()
    X=X.diff()*100
    X=X.loc['26/10/2020  3:00:00 AM':]

    matrix_graph_box={}
    i=0
    for hedge_pt in hedge_box:
        pd_hedge=X.loc[:,hedge_pt]
        np_hedge=np.asarray(pd_hedge)
        hedge=np.zeros((len(X.columns),len(pd_hedge.columns)))
        row_count=0
        for column in X: #range(X.shape[0]):
            pd_target=X.loc[:,column]
            np_target=np.asarray(pd_target)

            reg=LinearRegression().fit(np_hedge,np_target)

            coeff=reg.coef_
            sum_coeff=coeff.sum()
            coeff_norm=coeff/sum_coeff
            for pillar_count,weight_value in enumerate(coeff_norm):
                hedge[row_count,pillar_count]=weight_value
            row_count=row_count+1
        pd_hedge=pd.DataFrame(hedge,columns=[i for i in pd_hedge.columns])
        pd_hedge.insert(0, 'Target', [i for i in raw_data.columns])

        matrix_graph_box[i]=pd_hedge
        
        i=i+1
    return matrix_graph_box





        

def BackTest(curr,data:DataParam,model:Modelparam):
    '''Data Cleaning'''
    raw_data = pd.read_excel('//SVRSG001RPS01.asia.corp.anz.com/huy11$/My Documents/project/DataDownload.xlsm',sheet_name=curr,index_col='Dates')
    target, hedge_pt, window_size=data.target, data.hedge, data.window_size
    
    period = model.period.value

    raw_data=raw_data.iloc[:,0:19]
    raw_data=raw_data.dropna()
    X=raw_data.copy()
    X=X.diff()*100

    '''Target and Hedge points Settings'''
    X=X.loc['26/10/2020  3:00:00 AM':]
    pd_hedge=X.loc[:,hedge_pt]
    np_hedge=np.asarray(pd_hedge)
    pd_target=X.loc[:,[target]]
    np_target=np.asarray(pd_target)
    
    timestamp_all=pd_hedge.index.tolist()

    '''Period Setting'''
    data_len=np_hedge.shape[0]
    test=data_len-window_size 
    model_error_matrix=[]
    
    '''Window Loop'''
    coeff_norm_box=[]
    time_stamp_box=[]
    
    for i in range(0,(test-period),period):
        time_stamp_part = timestamp_all[window_size+i]
        time_stamp_box.append(time_stamp_part)
        '''Train with Part of Window'''
        X_train=np_hedge[i:window_size+i,:]
        Y_train = np_target[i:window_size+i,:]

        '''Choose Model to Get Regression'''
        regression = []
        if model.type=='Linear': regression = getLR(X_train, Y_train, model)  
        if model.type=='PCA': regression = getPCA(X_train, Y_train, model)


        '''Normalization'''
        sum_coeff=regression.sum()
        coeff_norm=regression/sum_coeff
        coeff_norm_box.append(coeff_norm)
        np_weight=coeff_norm.T
        
        

        
        Y_test = np_target[window_size+i:window_size+i+period,:]
        X_test = np_hedge[window_size+i:window_size+i+period,:]
        

        '''Calculate the Test error'''
        model_error=np.squeeze(Y_test.T)-np.matmul(X_test,np_weight).T
        
        model_error_matrix.append(model_error.sum())
    
    weighted_beta_df=pd.DataFrame(np.squeeze(coeff_norm_box), columns=[i for i in pd_hedge.columns])

    
    
    returns={}
    returns['weighted_beta']=weighted_beta_df
    
    
   
    returns['timestamp']=time_stamp_box

    


    '''Plot Accumulative Error Graph'''
    model_error_cumsum=np.cumsum(model_error_matrix,axis=0)

    yvalues=model_error_cumsum
    
    returns['yvalues']=model_error_cumsum
    return returns