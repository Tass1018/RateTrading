from tkinter import *
from tkinter.ttk import *

################################################################################################################
#VARIABLES

CURRENCY = ('JPYT', 'USDS')
TARGET_YEAR = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50)
COMPARE_YEAR = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50)
SELECTED_YEARS = []


################################################################################################################
#FUNCTIONS
def selected_yrs(event):
    yrs = listbox_y.curselection()

    for yr in yrs:
        print("im here")
        selection = listbox_y.get(yr)
        SELECTED_YEARS.append(selection)
    for val in SELECTED_YEARS:
        print(val)
   



def confirm():
    cnf_curr  = combo_curr.get()
    cnf_target_yr = combo_x.get()
    cnf_compared_yrs = selected_yrs(SELECTED_YEARS)
    if len(SELECTED_YEARS)==0 or len(combo_curr.get())==0 or len(combo_x.get())==0:
        cnf_curr = ""
        cnf_target_yr = -1
        cnf_compared_yrs = []
        
        print("Information Incomplete!")
        return
    

def delete():
    combo_curr.set('')
    combo_x.set('')
    listbox_y.selection_clear(0, 'end')


def open_LR():
    window_lr = Toplevel(window)
    fig_lr = Label(window_lr, text="THIS IS A LINEAR RREGRESSION GRAPH:)", font=("Arial Bold", 15))
    fig_lr.grid(column=2, row=0) 


def open_PCA():
    window_pca = Toplevel(window)
    fig_pca = Label(window_pca, text="THIS IS A PCA GRAPH:)", font=("Arial Bold", 15))
    fig_pca.grid(column=2, row=0) 


def open_Coin():
    window_coin = Toplevel(window)
    fig_coin = Label(window_coin, text="THIS IS A COINTAGE GRAPH:)", font=("Arial Bold", 15))
    fig_coin.grid(column=2, row=0) 
################################################################################################################
 

#HEAD PAGE
window = Tk()
window.title("Swap Models For Various Currencies")
window.geometry('920x350')



lbl = Label(window, text="Swap Models For Various Currencies", font=("Arial Bold", 15))
lbl.grid(column=2, row=0)

lbl_currency = Label(window, text="CURRENCY", font=("Arial Bold", 10))
lbl_currency.grid(column=2, row=50)
combo_curr = Combobox(window, width=5)
combo_curr['values']= CURRENCY
combo_curr.grid(column=2, row=60)
combo_curr.get()



#TARGET YEAR
lbl_x = Label(window, text="TARGET YEAR", font=("Arial Bold", 10))
lbl_x.grid(column=2, row=70)
combo_x = Combobox(window, width=5)
combo_x['values']= TARGET_YEAR
combo_x.grid(column=2, row=80)
combo_x.get()



#COMPARING YEAR
lbl_y = Label(window, text="COMPARED YEARS", font=("Arial Bold", 10))
lbl_y.grid(column=2, row=90)
listbox_y = Listbox(window, height=3, selectmode=MULTIPLE)
j = 1
for i in COMPARE_YEAR:
    listbox_y.insert(j, i)
    j+=1
listbox_y.grid(column=2, row=100)



#BOTTONS
btn_confirm = Button(window, text='CONFIRM', width = 40, command=confirm)
btn_confirm.grid(column=1, row=150, padx=20, pady=40)
btn_clear = Button(window, text='CLEAR ALL', width = 40, command=delete)
btn_clear.grid(column=3, row=150)

btn_linearR = Button(window, text='Linear Regression', command=open_LR)
btn_linearR.grid(column=1, row=200)

btn_PCA = Button(window, text='PCA', command=open_PCA)
btn_PCA.grid(column=2, row=200)

btn_cointegration = Button(window, text='Cointigration', command=open_Coin)
btn_cointegration.grid(column=3, row=200)




window.mainloop()








