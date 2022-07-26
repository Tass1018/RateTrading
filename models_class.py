from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import ipywidgets as widgets
import pca_sofr
import pca_inputs
from pca_sofr import GetPCA
from lr_model import getLR
from lr_input import plot_lr
from pca_inputs import plotPCA
from ipywidgets import interact, interactive, fixed, interact_manual
from ipywidgets import HBox, VBox, GridBox, Layout
from IPython.display import display
from ipywidgets import Label, Layout, HBox
from IPython.display import display

class DataParam:
    def __init__(self):
        self.target_index =widgets.IntText(
            value=6,
            description='Target Points:',
            style={'description_width': 'initial'},
            layout = widgets.Layout(width='200px')
        )
        self.pillar_textbox =widgets.Text(
            value='2, 5, 10',
            description='Hedge Points:',
            style={'description_width': 'initial'},
            layout = widgets.Layout(width='200px')
        )
        
        self.window_index =widgets.IntText(
            value=60,
            description='Hist Window:',
            style={'description_width': 'initial'},
            layout = widgets.Layout(width='200px')
        )
        
        self.index =widgets.Text(
            value='',
            description='ModelID',
            style={'description_width': 'initial'},
            layout = widgets.Layout(width='150px')
        )
        
        
        
        self.number_chart_list = []
        if self.pillar_textbox.value is not None:  
            chart_list=[i for i in self.pillar_textbox.value.split(',')]
            for num in chart_list:
                self.number_chart_list.append(int(num))
                
                
        self.target = self.target_index.value
        self.hedge = self.number_chart_list
        self.window_size = self.window_index.value
        self.i = self.index.value
        
        line=HBox(children=[self.target_index, self.pillar_textbox, self.window_index, self.index])
        display(line)
    
    def Update(self,modeldic):
        #print("i am updating")
        self.number_chart_list=[]
        if self.pillar_textbox.value is not None:  
            chart_list=[i for i in self.pillar_textbox.value.split(',')]
            for num in chart_list:
                self.number_chart_list.append(int(num))
                
        self.target = self.target_index.value
        self.hedge = self.number_chart_list
        self.window_size = self.window_index.value
        self.i = self.index.value
        
        
        
    

class Modelparam:
    def __init__(self, index):
        self.dropdown_models=widgets.Dropdown(
            options=['Linear', 'PCA'],
            description='Regression:',
            style={'description_width': 'initial'},
            layout = widgets.Layout(width='200px')
        )

        self.period =widgets.IntText(
            value=2,
            description='Period:',
            style={'description_width': 'initial'},
            layout = widgets.Layout(width='200px')
        )

        self.number_of_components =widgets.IntText(
            value=3,
            description='Num_of_Components:',
            style={'description_width': 'initial'},
            layout = widgets.Layout(width='200px')
        )

        self.constrain_formula =widgets.Dropdown(
            options=['None', 'Lasso', 'Ridge', 'Both'],
            description='Loss Function:',
            style={'description_width': 'initial'},
            layout = widgets.Layout(width='200px')
        )
        self.lamb_index1 =widgets.FloatText(
            value=0,
            description='Lasso Lambda:',
            style={'description_width': 'initial'},
            layout = widgets.Layout(width='200px')
        )
        self.lamb_index2 =widgets.FloatText(
            value=0,
            description='Ridge Lambda:',
            style={'description_width': 'initial'},
            layout = widgets.Layout(width='200px')
        )
        

        box_layout = Layout(grid_template_columns="repeat(3, 250px)",
                            
                                    flex_flow='column',
                                    align_items='stretch',
                                    border='solid 3px #6297bf',
                                    width='100%')
        self.name = index
        self.period_index = self.period.value
        self.number_of_components_index = self.number_of_components.value
        self.constrain_formula_index = self.constrain_formula.value
        self.lamb_index_index1 = self.lamb_index1.value
        self.lamb_index_index2 = self.lamb_index2.value
        self.type = self.dropdown_models.value
        
        z = Label("Model ID"+self.name, layout=Layout(display="flex", justify_content="flex-end", width="30%", border="solid"))
        
        
        line=GridBox(children=[self.dropdown_models, self.period, self.number_of_components, self.lamb_index1,self.lamb_index2,z], layout=box_layout)
        display(line)

    def Update(self): 
        self.name = self.name
        self.period_index = self.period.value
        self.number_of_components_index = self.number_of_components.value
        self.constrain_formula_index = self.constrain_formula.value
#         print(self.constrain_formula_index)
        self.lamb_index_index1 = self.lamb_index1.value
        self.lamb_index_index2 = self.lamb_index2.value
        self.type = self.dropdown_models.value
    def GetName(self):
        return self.name
