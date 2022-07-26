import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from scipy import stats

def plot_lr(curr, pillars, spot_swap):
    file_dir='//SVRSG001RPS01.asia.corp.anz.com/huy11$/My Documents/project/'

    """use pandas to import from excel file"""
    raw_data=pd.read_excel(file_dir+'DataDownload.xlsm',sheet_name=curr,index_col='Dates')
    print(raw_data.shape)

    raw_data=raw_data.iloc[:,0:16]
    raw_data=raw_data.dropna()
    X=raw_data.copy()
    X=X.diff()
    X=X.between_time('4:00:00','18:00:00')
    X=X.loc['26/10/2020':]
    x_values=X.index.values

    swap_points=[i for i in X.iloc[:,0:16].columns]


    pd_tempmtx=X.loc[:,pillars]
    pd_tempcol=X.loc[:,[spot_swap]]


    np_tempmtx=np.asarray(pd_tempmtx)
    np_tempcol=np.asarray(pd_tempcol)


    reg=LinearRegression().fit(np_tempmtx,np_tempcol)
    coeff=reg.coef_

    sum_coeff=coeff.sum()

    coeff_norm=coeff/sum_coeff

    sum_coeff_norm=coeff_norm.sum()


    np_weight=coeff_norm.transpose()

    model_Y=np.matmul(np_tempmtx,np_weight)

    model_error=np_tempcol-model_Y

    model_error_cumsum=100*np.cumsum(model_error,axis=0)
    print(model_error_cumsum)
    yvalues=model_error_cumsum

    zscore=stats.zscore(yvalues)
    mean=[]
    for i in range(len(yvalues)):
        mean.append(np.mean(yvalues))


    plt.xlabel('date')
    plt.ylabel('residual error')
    plt.grid(True)
    plt.plot(yvalues)
    plt.plot(zscore, label='Zscore', color="green")
    plt.plot(mean, label='Mean', linestyle='--')
    plt.show()
