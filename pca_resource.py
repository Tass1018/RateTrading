import matplotlib.pyplot as plt
from pylab import *

class PCAResults:
    """
    Class to save down PnL, hedge ratio results (amongst others) after running analysis in order to
    chart the results at a later time.  The results will be saved down from a full analysis and will
    be accessible by loading the saved object and accessing a data point by referencing the index
    from the pandas file which is saved down as a csv at the time of the analysis
    """
    
    def __init__(self, *args):
        """
        args is list representing name of parameter to save-down (used for accessing data)
        _data_type_dict: internal dictionary that assigns number to each data input item
        data_dict: dictionary which stores all the data
        """
        self._data_type_dict={i:j for (i,j) in enumerate(args)}
        self._inv_data_type_dict={j:i for i,j in self._data_type_dict.items()}
        self.data_dict={}

    def SaveData(self,data_point,data_list,data_type):
        """
        data_point: a number which dictates which iteration of the PCA is being referenced 
        (equivalent to the index value saved down in excel file in pca sofr.py module)
        data_type: the data type to attribute data to (must have been added when class instantiated) 
        data_list: python list of data to be saved
        """
        
        if data_point not in self.data_dict.keys():
            self.data_dict[data_point]={k:None for k in self._data_type_dict.keys()}
        datatype=self._inv_data_type_dict[data_type]
        self.data_dict[data_point][datatype]=data_list

    def DisplayDataPoints(self):
        """
        Output the names of categories of data saved in file
        """
        [print(k,':',v) for k,v in self._data_type_dict.items()]

    def ChartData(self,index,inputs):
        """
        charting tool for object data
        index: key from data_dict that requires charting
        inputs: python list input of data to be charted.  New subplot for each item in list
       """
        data_to_plot=len(inputs)
        for i,v in enumerate(range(data_to_plot)):
            if inputs[i] is None:
                print("its none haha1")
                inputs[i]=0
            if v is None:
                print("its none haha2")
                v=0
            if self.data_dict[int(index)] is None:
                print("its none haha3")
                self.data_dict[int(index)][int(inputs[i])]=0

            if int(inputs[i]) in self.data_dict[int(index)].keys():
                v = v+1
                ax1 = subplot(data_to_plot,1,v)

                ax1.plot(self.data_dict[int(index)][int(inputs[i])])#,label='{}'.format(self._data_type_dict[i]))
        # pickle.dump(ax1, file('myplot.pickle', 'w'))
        plt.show()



