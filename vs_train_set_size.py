#%%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utility.plot_utility_v3 import scatter_plot, font_legend

"""
module_root = os.path.dirname(os.path.realpath('__file__'))
sys.path.append(module_root + "\\MolGraph")
"""
from gd_dpo.mod_GDDPO import modGD_DPO
from gd_dpo.GDDPO import GD_DPO
from utility.stratified_splitting import sampling
from trainer import train_model,pearson_corr,RMSD

total = pd.read_csv(
    "total_data.csv"
    #"total_data_plus_pah_n_thieno.csv"
    )
total = total.fillna(0)
plist = [
    "s","a","b","c","d"
        ]

intervals = [1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]

avg_errors = 0
all_errors = []
len_trains = []

train_sizes = [0.01*num for num in range(1,11,2)]
train_sizes += [0.1*num for num in range(1,11,1)]
train_set_size_list = [round(0.6*len(total)*size) for size in train_sizes]

N = 10
np.random.seed(906)
seed_list1 = np.random.randint(0,10e6,N)
seed_list2 = np.random.randint(0,10e5,N)

for j in range(N):
    errors = []
    train_set_complement = sampling(
        round(0.6*len(total)),
        total, 
        intervals = intervals,
        random_state=seed_list1[j]
    )

    test = total.drop(index = train_set_complement.index,axis=0)

    DPO_test = test.loc[:,"DPO"]
    SDPO_test = test.loc[:,"SDPO"]
    
    train_set = pd.DataFrame(columns = total.columns)
    
    for i,train_set_size in enumerate(train_set_size_list):
        sample_size = train_set_size-train_set_size_list[i-1] if i>0 else train_set_size

        train_set_add = sampling(
            sample_size,train_set_complement,
            intervals = intervals,
            random_state = seed_list2[j])
        
        #add the sampled data to the existing training set
        train_set = pd.concat([train_set,train_set_add])
        
        #for examining what are those training sets
        #drop the sampled data from the training reservoir
        train_set_complement = train_set_complement.drop(
            train_set_add.index)
        
        DPO = train_set.loc[:,"DPO"]
        SDPO = train_set.loc[:,"SDPO"]

        Y1 = np.array(train_set.loc[:,"BG"])
        Y2 = np.array(train_set.loc[:,"EA"])
        Y3 = np.array(train_set.loc[:,"IP"])

        model_mt = modGD_DPO(
            parameters_list=plist,
            parameter_values=list(np.random.rand(len(plist))/10),
            constant_list= ["sa","b0","bf","cf"],
            constant_values=[1,0,0,0],
            tasks = 3
        )

        epochs = 150
        lr = 1

        model_mt = train_model(
            model_mt,
            DPO, SDPO, Y1,Y2, Y3,
            epochs,lr
            )
        
        len_trains.append(len(DPO))
        
        for i,elec_prop in enumerate(["BG","EA","IP"]):
            Y_test = np.array(test.loc[:,elec_prop])
            
            _,_,Y_hat = model_mt.predict(DPO_test,SDPO_test,i)
            
            errors.append(RMSD(Y_test,Y_hat))

    errors = np.array(errors).reshape(-1,3)
    all_errors.append(errors)
        

# %%
avg_errors = np.mean(all_errors,axis=0)
error_std = np.std(all_errors,axis=0)

major_ticks = [0.02,0.05,0.05]

ylabel = ["bandgap","electron affinity","ionization potential"]

label_coord = [
    [580, 0.28],
    [580,0.35],
    [580,0.45]
]

labels = ["(A)","(B)","(C)"]


for i,elec_prop in enumerate(["BG","EA","IP"]):
    plot = scatter_plot()

    plot.add_plot(
        train_set_size_list,
        errors[:,i],
        plot_line = True,
        x_major_tick=50,
        x_minor_tick=10,
        xticks_format = 0 if i == 2 else -1,
        
        y_major_tick=major_ticks[i],
        y_minor_tick=0.01,
        yticks_format=2,
        
        xlabel = "Training set sizes (samples)" if i == 2 else None,
        ylabel = "RMSD for {}(eV)".format(ylabel[i])
    )
    
    plot.ax.errorbar(
        train_set_size_list,
        errors[:,i],
        error_std[:,i], 
        color = "black",
        capsize = 2.0,
        fmt = 'none'
        )

    
    plot.add_text2(
        0.9,0.9,labels[i])
    
    plot.save_fig("_rmsd_vs_train_size_"+elec_prop+".jpeg")
# %%
for i,size in enumerate(train_set_size_list):
    print(size,avg_errors[i,0:3],error_std[i,0:3])
# %%
