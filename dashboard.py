from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from ipywidgets import HBox, VBox
from IPython.display import display
from ipywidgets import Output, Tab

from models_class import DataParam
from models_class import Modelparam
from backtest import BackTest,MidBackTest

from random import seed
from random import randint

import plotly.express as px
import plotly.graph_objects as go


'Data'
CURRENCY = pd.ExcelFile('//SVRSG001RPS01.asia.corp.anz.com/huy11$/My Documents/project/DataDownload.xlsm').sheet_names[1:7]
from random import seed
from random import randint
menu = ['structure','daily_pnl','total_pnl','hedge_ratio_1','hedge_ratio_2','hedge_ratio_3','hedge_ratio_4']

###########################################################################################################################################
'widgets'
dropdown_curr = widgets.Dropdown(options=CURRENCY, description='Currency:',layout = widgets.Layout(width='200px'))

###########################################################################################################################################
'bottons'
"Add/Remove Model"
index = 0
mode_dic={}
model_storage = []

def add_model():
    out = Output()
    with out:
        m = Modelparam(randint(0,100).__str__())
        model_storage.append(m)
       
    return out

model_box = widgets.VBox([add_model()])
btn_add_model = widgets.Button(description = 'Add Model') 

def remove_model():
    rem = Output()
    with rem:
        remove = model_box.children[-1]
        model_box.children = model_box.children[:-1]
        remove.close()
        model_storage.pop()
        mode_dic.popitem()
    return rem
btn_rem_model = widgets.Button(description = 'Remove Model') 

"Save"
button_save = widgets.Button(
    description='OK',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='OK',
    layout = widgets.Layout(width='300px'),
    icon='' 
)

"Add/Remove Data"
data_storage = []
def add():
    out = Output()
    with out:
        m = DataParam()
        data_storage.append(m)
        
    return out

data_box = widgets.VBox([add()])
btn_add = widgets.Button(description = 'Add Scenario') 

def remove():
    rem = Output()
    with rem:
        remove = data_box.children[-1]
        data_box.children = data_box.children[:-1]
        remove.close()
        data_storage.pop()
        
    return rem
btn_rem = widgets.Button(description = 'Remove Scenario') 


    

"Run"
button_run1 = widgets.Button(
    description='Run',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Run',
    layout = widgets.Layout(width='300px'),
    icon='' 
)



'Dashboard'
Add_remove_model = HBox([btn_add_model, btn_rem_model, button_save])
Add_remove_data = HBox([btn_add, btn_rem])
dashboard = widgets.VBox([dropdown_curr, model_box, Add_remove_model, data_box, Add_remove_data, button_run1])

    

###########################################################################################################################################
'Interactions'

def botton_addm_on_click(a):
    model_box.children=tuple(list(model_box.children) + [add_model()])
    return
btn_add_model.on_click(botton_addm_on_click)
index=index+1

def botton_remm_on_click(a):
    remove_model()
    return
btn_rem_model.on_click(botton_remm_on_click)

def botton_save_on_click(b):
    for model in model_storage:
        if model is None:
            print("model is none")
        else:
            model.Update()
            mode_dic[model.name]=model
#             print(model.name)
            
               
button_save.on_click(botton_save_on_click)


def botton_run1_on_click(b):
    residual_error_dic={}
    hedgeratio_dic={}
    timestamp_dic={}
    hedge_box=[]
    for data in data_storage:
        data.Update(mode_dic)
        name=data.target.__str__()+'y vs '+format(data.hedge)+ '[MOD'+data.i+']'
        if data.hedge.__str__() not in hedge_box:
            hedge_box.append(data.hedge)
        backtest=BackTest(dropdown_curr.value, data, mode_dic[data.index.value])
        residual_error_dic[name] = backtest['yvalues']
        hedgeratio_dic[name] = backtest['weighted_beta']
        timestamp_dic[name] = backtest['timestamp']
        
    
    matrix_box = MidBackTest(dropdown_curr.value, hedge_box)
    
    plotTable(residual_error_dic)
    
    plotResidualError(residual_error_dic,timestamp_dic)
    
    plotHedgeRatio(hedgeratio_dic, timestamp_dic)
    
    plotMatrix(matrix_box)


        
button_run1.on_click(botton_run1_on_click)


    
def botton_add_on_click(a):
    data_box.children=tuple(list(data_box.children) + [add()])
    return
btn_add.on_click(botton_add_on_click)

def botton_rem_on_click(a):
    remove()
    return
btn_rem.on_click(botton_rem_on_click)



#########################################################################################################
"Graphing"
def plotTable(residual_error_dic):
    dfff=plotSummary(residual_error_dic)

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(dfff.columns),
                    fill_color='lightskyblue',
                    align='center'),
        cells=dict(values=[dfff['Name'], dfff['Mean'], dfff['Range'], dfff['Std'], dfff["0~0.05"]],
                fill_color='lavender',
                align='center'))
    ])
    fig.show()
    
    
import statistics
def plotSummary(residual_error_dic):
    names = residual_error_dic.keys()
    SUMMARY = pd.DataFrame(columns=["Name", "Mean", "Range", "Std","Max Dropdown(weekly)","Max Dropdown(monthly)","0~0.05"])
    for name in names:
        df=residual_error_dic[name]
#         df=df[:,0]
        std=round(df.std(),2)
        mean=round(df.mean(),2)
        max=round(df.max(),2)
        min=round(df.min(),2)
        range = '['+min.__str__() + ' , ' + max.__str__() + ']'
        values = df
        count=0
        for val in values:
            if val <= 0.05 and val >= -0.05:
                count=count+1

        P_range=round(count/len(values)*100,2)
        P=P_range.__str__()+'%'

        dict = {'Name': name, 'Mean': mean, 'Range': range,'Std': std, "0~0.05": P}
        SUMMARY = pd.concat([SUMMARY, pd.DataFrame.from_records([dict])])

    return SUMMARY


def plotResidualError(residual_error_dic, timestamp_dic):
    df = pd.DataFrame(residual_error_dic)
    df.index= list(timestamp_dic.items())[0][1]
    graphs=[]
    
    for scenario in df.columns:
        graphs.append(df.loc[:,scenario])
        
        
    fig = px.line(df, title="Residual Error")
    fig.update_xaxes(
                title_text = "Date",
                title_font = {"size": 15},
                title_standoff = 25)

    fig.update_yaxes(
            title_text = "Residual Error Index",
            title_font = {"size": 15},
            title_standoff = 25)
    
    figw = go.FigureWidget(fig)
    fig.show()

    
    
def plotMatrix(matrix_box):
    fig_box=[]
    for i in matrix_box:
        
        pd_hedge=matrix_box[i].round(2)
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(pd_hedge.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[pd_hedge[col] for col in pd_hedge.columns],
                       fill_color='lavender',
                       align='left'))
        ], layout=layout)
        
        figw = go.FigureWidget(fig)
        fig_box.append(figw)
    container = HBox(fig_box)
    display(container) 
    
def plotHedgeRatio(hedgeratio_dic, timestamp_dic):
    keys = hedgeratio_dic.keys()
    for key in keys:
        df=hedgeratio_dic[key]
        df.index=timestamp_dic[key]
        
        fig = px.line(df, title="Hedge Ratio of "+ key)
        fig.update_xaxes(
                title_text = "Date",
                title_font = {"size": 15},
                title_standoff = 25)

        fig.update_yaxes(
                title_text = "Hedge Ratio Index",
                title_font = {"size": 15},
                title_standoff = 25)
        fig.show()
        


    
        
        
layout = go.Layout(
    autosize=False,
    width=400,
    height=600
)  

 
def format(hedge):
    str=''
    for i in range(len(hedge)):
        if i==len(hedge)-1:
            str=str+hedge[i].__str__()+'y '
        else:     
            str=str+hedge[i].__str__()+'y,'
    return str
