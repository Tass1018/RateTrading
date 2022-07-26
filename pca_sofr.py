import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pca_resource import PCAResults
import pickle

'select parameters for analysis'
def GetPCA(curr, pt):

    test_pillars=[pt]
    window=[i for i in range(60,500,30)] #range(start, stop, step)
    test_period=[2] 

    lall=[1,2,3,4,5,6,7,8,9,10,12,15,20,25,30]
    l0=[2,3,5,10,30]
    l1=[4,7,15,20]
    l2=[6,8,9,12,25]

    result_df=pd.DataFrame(columns=['hist_window','test_window','strategy','mean','vol','max_dd','abs_mean'])

    # file_dir='C:/NotBackedUp/'
    # file_dir_obj='\\SVRSG001RPS01.asia.corp.anz.com\huy11$\My Documents\project\'
    data_store={}

    raw_data=pd.read_excel('//SVRSG001RPS01.asia.corp.anz.com/huy11$/My Documents/project/DataDownload.xlsm',sheet_name=curr,index_col='Dates')
    raw_data=raw_data.iloc[:,0:19]
    X=raw_data.copy()

    X=X.diff()*100

    for pillar in test_pillars:

        data_store[pillar]=PCAResults('structure','daily_pnl','total_pnl','hedge_ratio_1','hedge_ratio_2','hedge_ratio_3','hedge_ratio_4')        
        spreads=[]
        flys=[]
        box=[]
        input_count=0

        if pillar in l1:
            input=l0
        else:
            input=l0
            input.extend(l1)
    

        for i in input:
            for j in input:
                if j>i: spreads.append((i,j))
        for i in input:
            for j in input:
                for k in input:
                    if j>i and k>j: flys.append((i,j,k))
        for i in input:
            for j in input:
                for k in input:
                    for l in input:
                        if j>i and k>j and l>k: box.append((i,j,k,l))
        for window_size in window: # 60, 90, 120, 150, ..., 600

            for period in test_period:

                print('Calculating swap point: {}, window size: {}, period: {}'.format(pillar,window_size,period), end='\r')

                for strat in spreads: #Q: leg3 has to fall between leg1 and leg2?
                    minimum=99
                    leg1,leg2,leg3=lall.index(strat[0]),lall.index(strat[1]),lall.index(pillar)                
                    if leg1==2 and leg2==3: # consecutive?
                        continue
                    if leg3<leg1: #can't predict the yr with longer term rate in the future?
                        continue
                    if leg3<9 and leg1 in l1 and leg1<9 and (leg3-leg1)>1:
                        continue
                    if leg3<9 and leg2 in l1 and leg2<9 and (leg3-leg2)>1:
                        continue
                    
                    X_all=np.array(X[[strat[0],strat[1],pillar]].dropna())
                    data_len=X_all.shape[0]
                    test=data_len-window_size 
                    estimate_count=int((test-period)/period)
                    mean_total=0
                    sigma_total=0
                    max_dd=0
                    total_pnl=[]
                    daily_pnl=[]
                    hedge_ratio_1=[]
                    hedge_ratio_2=[]
                    min_daily_pnl=99
            

                    for i in range(0,(test-period),period): #why not test?
                        X_train_all=X_all[i:window_size+i,:]
                        X_test=X_all[window_size+i:window_size+i+period,:]
                        X_ctd=X_train_all-np.mean(X_train_all,axis=0)
                        X_covar=np.cov(X_ctd,rowvar=False)
                        u,s,vt=np.linalg.svd(X_covar)
                        e_values=np.array([s[i]**2 for i in range(s.shape[0])])
                        e_vectors=vt.T[:,:2]
                        PC=np.matmul(np.linalg.inv(e_vectors[:-1,:]),X_test[:,:-1].T)
                        y_pred=np.matmul(e_vectors[-1,:],PC)
                        
                        hedge_ratio=np.squeeze(np.matmul(e_vectors[-1,:],np.linalg.inv(e_vectors[:-1,:])))
                        results=y_pred-np.squeeze(X_test[:,-1])
                        mean,sigma,dd=np.mean(results),np.std(results),np.min(results)
                        mean_total+=mean/estimate_count
                        sigma_total+=sigma/estimate_count
                        daily_pnl.append(np.sum(results))
                        if i==0: total_pnl.append(daily_pnl[-1])
                        else: total_pnl.append(daily_pnl[-1]+total_pnl[-1])
                        hedge_ratio_1.append(hedge_ratio[0])
                        hedge_ratio_2.append(hedge_ratio[1])
                        if dd<max_dd: max_dd=dd


                    for i in range(0, len(daily_pnl)-10, 1):
                        pnl = sum(daily_pnl[i:i+10])
                        if pnl < min_daily_pnl:
                            min_daily_pnl = pnl


                    data_store[pillar].SaveData(input_count,strat,data_type='structure')
                    data_store[pillar].SaveData(input_count,hedge_ratio_1,data_type='hedge_ratio_1')
                    data_store[pillar].SaveData(input_count,hedge_ratio_2,data_type='hedge_ratio_2')
                    data_store[pillar].SaveData(input_count,daily_pnl,data_type='daily_pnl')
                    data_store[pillar].SaveData(input_count,total_pnl,data_type='total_pnl')
                    result_df=result_df.append({'hist_window':window_size,'test_window':period,'strategy':strat,'mean':mean,'vol':sigma,'max_dd':max_dd,'abs_mean':np.abs(mean), 'DROP':min_daily_pnl},ignore_index=True)
                    input_count+=1
                
                for strat in flys:
                   
                    leg1,leg2,leg3,leg4=lall.index(strat[0]),lall.index(strat[1]),lall.index(strat[2]),lall.index(pillar) 
                    if leg1==2 and leg2==3:
                        continue
                    if leg4<leg1:
                        continue
                    if leg4<9 and leg1 in l1 and leg1<9 and (leg4-leg1)>1:
                        continue
                    if leg4<9 and leg2 in l1 and leg2<9 and (leg4-leg2)>1:
                        continue
                    if leg4<9 and leg3 in l1 and leg3<9 and (leg4-leg3)>1:
                        continue     
                    X_all=np.array(X[[strat[0],strat[1],strat[2],pillar]].dropna())
                    data_len=X_all.shape[0]
                    test=data_len-window_size
                    estimate_count=int((test-period)/period)
                    mean_total=0
                    sigma_total=0
                    max_dd=0
                    total_pnl=[]
                    daily_pnl=[]
                    hedge_ratio_1=[]
                    hedge_ratio_2=[]
                    hedge_ratio_3=[]
                    min_daily_pnl=99
                    error=[]

                    for i in range(0,test-period,period):
                        X_train_all=X_all[i:window_size+i,:]
                        X_test=X_all[window_size+i:window_size+i+period,:]
                        X_ctd=X_train_all-np.mean(X_train_all,axis=0)
                        X_covar=np.cov(X_ctd,rowvar=False)
                        u,s,vt=np.linalg.svd(X_covar)
                        e_values=np.array([s[i]**2 for i in range(s.shape[0])])
                        e_vectors=vt.T[:,:3]
                        PC=np.matmul(np.linalg.inv(e_vectors[:-1,:]),X_test[:,:-1].T)
                        y_pred=np.matmul(e_vectors[-1,:],PC)
                        hedge_ratio=np.squeeze(np.matmul(e_vectors[-1,:],np.linalg.inv(e_vectors[:-1,:])))
                        results=y_pred-np.squeeze(X_test[:,-1])
                        mean,sigma,dd=np.mean(results),np.std(results),np.min(results)
                        mean_total+=mean/estimate_count
                        sigma_total+=sigma/estimate_count
                        daily_pnl.append(np.sum(results))
                        if i==0: total_pnl.append(daily_pnl[-1])
                        else: total_pnl.append(daily_pnl[-1]+total_pnl[-1])
                        hedge_ratio_1.append(hedge_ratio[0])
                        hedge_ratio_2.append(hedge_ratio[1])
                        hedge_ratio_3.append(hedge_ratio[2])
                        model_y = np.matmul(hedge_ratio, )
                        if dd<max_dd: max_dd=dd
                    
                    for i in range(0, len(daily_pnl)-10, 1):
                        pnl = sum(daily_pnl[i:i+10])
                        if pnl < min_daily_pnl:
                            min_daily_pnl = pnl
                    data_store[pillar].SaveData(input_count,strat,data_type='structure')
                    data_store[pillar].SaveData(input_count,hedge_ratio_1,data_type='hedge_ratio_1')
                    data_store[pillar].SaveData(input_count,hedge_ratio_2,data_type='hedge_ratio_2')
                    data_store[pillar].SaveData(input_count,hedge_ratio_3,data_type='hedge_ratio_3')
                    data_store[pillar].SaveData(input_count,daily_pnl,data_type='daily_pnl')
                    data_store[pillar].SaveData(input_count,total_pnl,data_type='total_pnl')
                    result_df=result_df.append({'hist_window':window_size,'test_window':period,'strategy':strat,'mean':mean,'vol':sigma,'max_dd':max_dd,'abs_mean':np.abs(mean), 'DROP':min_daily_pnl},ignore_index=True)
                    input_count+=1

                
                for strat in box:
                    
                    leg1,leg2,leg3,leg4,leg5=lall.index(strat[0]),lall.index(strat[1]),lall.index(strat[2]),lall.index(strat[3]),lall.index(pillar) 
                    if leg1==2 and leg2==3:
                        continue
                    if leg5<leg1:
                        continue
                    if leg5<9 and leg1 in l1 and leg1<9 and (leg5-leg1)>1:
                        continue
                    if leg5<9 and leg2 in l1 and leg2<9 and (leg5-leg2)>1:
                        continue
                    if leg5<9 and leg3 in l1 and leg3<9 and (leg5-leg3)>1:
                        continue  
                    if leg5<9 and leg4 in l1 and leg4<9 and (leg5-leg4)>1:
                        continue  
                    
                    X_all=np.array(X[[strat[0],strat[1],strat[2],strat[3],pillar]].dropna())
                    data_len=X_all.shape[0]
                    test=data_len-window_size
                    estimate_count=int((test-period)/period)
                    mean_total=0
                    sigma_total=0
                    max_dd=0
                    total_pnl=[]
                    daily_pnl=[]
                    hedge_ratio_1=[]
                    hedge_ratio_2=[]
                    hedge_ratio_3=[]
                    hedge_ratio_4=[]
                    min_daily_pnl=99

                    for i in range(0,test-period,period):
                        X_train_all=X_all[i:window_size+i,:]
                        X_test=X_all[window_size+i:window_size+i+period,:]
                        X_ctd=X_train_all-np.mean(X_train_all,axis=0)
                        X_covar=np.cov(X_ctd,rowvar=False)
                        u,s,vt=np.linalg.svd(X_covar)
                        e_values=np.array([s[i]**2 for i in range(s.shape[0])])
                        e_vectors=vt.T[:,:4]
                        PC=np.matmul(np.linalg.inv(e_vectors[:-1,:]),X_test[:,:-1].T)
                        y_pred=np.matmul(e_vectors[-1,:],PC)
                        hedge_ratio=np.squeeze(np.matmul(e_vectors[-1,:],np.linalg.inv(e_vectors[:-1,:])))
                        results=y_pred-np.squeeze(X_test[:,-1])
                        mean,sigma,dd=np.mean(results),np.std(results),np.min(results)
                        
                        mean_total+=mean/estimate_count
                        sigma_total+=sigma/estimate_count
                        daily_pnl.append(np.sum(results))
                        if i==0: total_pnl.append(daily_pnl[-1])
                        else: total_pnl.append(daily_pnl[-1]+total_pnl[-1])
                        hedge_ratio_1.append(hedge_ratio[0])
                        hedge_ratio_2.append(hedge_ratio[1])
                        hedge_ratio_3.append(hedge_ratio[2])
                        hedge_ratio_4.append(hedge_ratio[3])
                        if dd<max_dd: max_dd=dd

                    for i in range(0, len(daily_pnl)-10, 1):
                        pnl = sum(daily_pnl[i:i+10])
                        if pnl < min_daily_pnl:
                            min_daily_pnl = pnl

                    data_store[pillar].SaveData(input_count,strat,data_type='structure')
                    data_store[pillar].SaveData(input_count,hedge_ratio_1,data_type='hedge_ratio_1')
                    data_store[pillar].SaveData(input_count,hedge_ratio_2,data_type='hedge_ratio_2')
                    data_store[pillar].SaveData(input_count,hedge_ratio_3,data_type='hedge_ratio_3')
                    data_store[pillar].SaveData(input_count,hedge_ratio_4,data_type='hedge_ratio_4')
                    data_store[pillar].SaveData(input_count,daily_pnl,data_type='daily_pnl')
                    data_store[pillar].SaveData(input_count,total_pnl,data_type='total_pnl')
                    result_df=result_df.append({'hist_window':window_size,'test_window':period,'strategy':strat,'mean':mean,'vol':sigma,'max_dd':max_dd,'abs_mean':np.abs(mean), 'DROP':min_daily_pnl},ignore_index=True)
                    input_count+=1

                    
        l0=[2,3,5,10,30]
        l1=[4,7,15,20]
        l2=[6,8,9,12,25]
        result_df.to_csv('//SVRSG001RPS01.asia.corp.anz.com/huy11$/My Documents/project/'+'{} pca results.csv'.format(pillar))
        result_df=None
        result_df=pd.DataFrame(columns=['hist_window','test_window','strategy','mean','vol','max_dd'])
        """save PCAResult file"""
        with open('{}.pickle'.format('//SVRSG001RPS01.asia.corp.anz.com/huy11$/My Documents/project/'+curr+str(pillar)), 'wb') as f:
            pickle.dump(data_store[pillar], f)

# GetPCA('USDS', 7)