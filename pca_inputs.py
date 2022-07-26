import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import pca_resource
import os

def plotPCA(pt, menu, index, curr):
    'select parameters for analysis'
    file_dir_obj='//SVRSG001RPS01.asia.corp.anz.com/huy11$/My Documents/project/'

    'ask user for swap point to analyse'
    print('The swap points are available:')
    for file in os.listdir(file_dir_obj):
        if file.endswith('.pickle'):
            print(file.split('.')[0])

    swap_pt=pt
    call_obj = open(file_dir_obj + curr + str(swap_pt)+'.pickle', "rb")
    swap_data=pickle.load(call_obj)

    """Print stored data categories"""
    print("Available data points")
    swap_data.DisplayDataPoints()

    """Request items to chart"""
    chart_data=menu
    chart_list=[i for i in chart_data.split(',')]

    """Initiate loop to analyse data"""
    while True:
        idx=index
        swap_data.ChartData(idx,chart_list)
        break