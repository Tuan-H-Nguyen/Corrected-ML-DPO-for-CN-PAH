#%%
import warnings
from matplotlib import projections
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
train = sampling(
    round(0.6*len(total)),
    total, 
    intervals = [1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0],
    random_state=949
    )

test = total.drop(index = train.index,axis=0)
print("training set size: {}, test set size: {}".format(len(train),len(test)))
dev_data = train

#dpo_type = ["Full_DPO", "Manual_SDPO"]
dpo_type = ["DPO","SDPO"]
sdpo_label = dpo_type[1]
dpo_label = dpo_type[0]

"""
plist = [
    "a","b","c","d"
        ] + ["sa"+str(i) for i in range(3,8)] + ["s","ds","sm"]
"""
plist = [
    "a","b","c","d"
        ] + ["s"]# + ["s","ds","sm"]


clist = [
    "sa",
    "b0","bf","cf"
    ]

cvalues = [1,0,0,0]

#%%
DPO = dev_data.loc[:,dpo_type[0]]
SDPO = dev_data.loc[:,dpo_type[1]]

Y1 = np.array(dev_data.loc[:,"BG"])
Y2 = np.array(dev_data.loc[:,"EA"])
Y3 = np.array(dev_data.loc[:,"IP"])


model_mt = modGD_DPO(
    parameters_list=plist,
    parameter_values=list(np.random.rand(len(plist))/10),
    constant_list= clist,
    constant_values=cvalues,
    tasks = 3
)

epochs = 50*3
lr = 1

model_mt, losses = train_model(
    model_mt,
    DPO, SDPO, Y1,Y2, Y3,
    epochs,lr, return_loss = True
    )

losses = np.array(losses)
rmsd = np.sqrt(losses)

rmsd = rmsd.reshape(-1,3)

epochs = np.array([i+1 for i in range(losses.shape[0])])
epochs = epochs.reshape(-1,3)
#%%
"""
###############
# CONVERGENCE #
###############
"""

plot = scatter_plot()

for i,elec_prop in enumerate(["Bandgap","EA","IP"]):
    plot.add_plot(
        epochs[:,1],rmsd[:,i],
        label = elec_prop,
        plot_line = True, 
        xlabel = "Time steps",
        ylabel = "Training RMSD for \nelectronic properties (eV)",
        xticks_format= 0
        )
    
plot.ax.legend()
plot.save_fig("convergence.jpeg",dpi=600)
# %%
"""
###########
#  TRAIN  #
###########
"""
#### IMPORTANT ####
save_fig = True
#### IMPORTANT ####

elec_prop_name = [
    "Band gap",
    "Electron Affinity",
    "Ionization Potential"
]

text_coord = [
    [0.98,0.80,0.98,0.80],
    [0.02,0.8,0.02,0.8],
    [0.02,0.8,0.98,0.8]
]
ha = [
    ["right","right"],
    ["left","left"],
    ["left","right"],
]

label = ["(A)","(B)","(C)"]

for i,elec_prop in enumerate(["BG","EA","IP"]):
    plot = scatter_plot()
    plot2 = scatter_plot()
    
    dpo = train.loc[:,dpo_label]
    sdpo = train.loc[:,sdpo_label]
    y = train.loc[:,elec_prop]
    y = np.array(y)
    
    xmt1,xmt2,_ = model_mt.predict(dpo,sdpo,i)
    
    wb = model_mt.outputs[i].weight[0][0]
    w1 = model_mt.outputs[i].weight[1][0][0]
    w2 = model_mt.outputs[i].weight[1][0][1]
    y_res = y - w1*xmt1.reshape(-1)
    
    plot.add_plot(
        xmt2, y_res, 
        scatter = False,
        plot_line = True,
        weight = (wb,w2),
    )
    
    plot2.add_plot(
        xmt1,y - w2*xmt2.reshape(-1),
        scatter = False,
        plot_line = True, 
        weight = (wb,w1),
    )
    
    print("Elec. prop.: {}".format(elec_prop))
    print("pearson DPO vs Y:",pearson_corr(xmt1,y)**2)
    
    pearson1 = pearson_corr(xmt2,y_res)**2
    pearson2 = pearson_corr(xmt1,y - w2*xmt2.reshape(-1))**2
    
    for no_sub in range(1,5):
        dpo = train.loc[train.no_subs == no_sub].loc[:,dpo_label]
        sdpo = train.loc[train.no_subs == no_sub].loc[:,sdpo_label]
        y = train.loc[train.no_subs == no_sub].loc[:,elec_prop]
        y = np.array(y)
        
        #model_mt.feedforward(dpo,sdpo,y,i)
        
        xmt1,xmt2,_ = model_mt.predict(dpo,sdpo,i)
        
        plot.add_plot(
            xmt2, y - w1*xmt1.reshape(-1),
            label = "{}-CN-group(s) PAHs".format(no_sub),
            xlabel = "Substituent descriptor" if i == 2 else None,
            ylabel = "{}\n after subtracting w$_1\\times$DPO(eV)".format(
                elec_prop_name[i]
            ),
            
            xticks_format=2 if i ==2 else -1,
            x_major_tick= 0.5,
            x_minor_tick= 0.1
        )
        if no_sub == 4:
            plot.add_text(
                text_coord[i][0],text_coord[i][1],
                "{}\nR$^2$ = {:.2f}".format(
                    label[i],pearson1),
                ha = ha[i][0]
            )
        
        if i == 1:
            plot.ax.legend(
                prop = font_legend,loc = "center left",
                bbox_to_anchor=(1.04,0.5), borderaxespad=0)
        
        if save_fig:
            plot.save_fig(elec_prop+"_excl_DPO_vs_SP.jpeg")
        
        plot2.add_plot(
            xmt1 , y - w2*xmt2.reshape(-1),
            xlabel = "Degree of $\pi$-orbital overlap" if i == 2 else None,
            ylabel = "{}\n after subtracting w$_2\\times$".format(elec_prop_name[i])+"S$_{DPO} $(eV)",
            label = "{}-CN-group(s) PAHs".format(no_sub),
            
            xticks_format=2 if i ==2 else -1,
            x_major_tick= 0.5,
            x_minor_tick= 0.1

        )
        if no_sub == 4:
            plot2.add_text(
                text_coord[i][2],text_coord[i][3],
                "{}\nR$^2$ = {:.2f}".format(
                    label[i],
                    pearson2),
                ha = ha[i][1]
            )
            
        if i == 1:
            plot2.ax.legend(
                prop = font_legend,loc = "center left",
                bbox_to_anchor=(1.04,0.5), borderaxespad=0)

        if save_fig:
            plot2.save_fig(elec_prop+"_excl_SP_vs_DPO.jpeg")


#%%
"""
###########
#  TEST   #
###########
"""
label = ["(A)","(B)","(C)"]

limit = [
    [1.0,4.5],
    [1.75,4.75],
    [4.5,7.25]
]

for i,elec_prop in enumerate(["BG","EA","IP"]):
    dpo = test.loc[:,dpo_type[0]]
    sdpo = test.loc[:,sdpo_label]

    xmt1,xmt2,y_hat = model_mt.predict(dpo,sdpo,i)

    y = np.array(test.loc[:,elec_prop]).reshape(-1)
    y_hat = y_hat.reshape(-1)
    
    glob_pearson = pearson_corr(y,y_hat)
    glob_rmsd = RMSD(y,y_hat)
    print("{}: RMSD = {:.2f}".format(
        elec_prop,glob_rmsd
    ))
    
    plot = scatter_plot()

    for no_sub in range(1,5):
        dpo = test.loc[test.no_subs == no_sub].loc[:,dpo_type[0]]
        sdpo = test.loc[test.no_subs == no_sub].loc[:,sdpo_label]
        
        xmt1,xmt2,y_hat = model_mt.predict(dpo,sdpo,i)

        y = test.loc[test.no_subs == no_sub].loc[:,elec_prop]
        y = np.array(y).reshape(-1)
        y_hat = y_hat.reshape(-1)
        
        print("# sub = {}, rmsd = {:.2f}eV".format(
            no_sub, RMSD(y,y_hat)
        ))
        
        plot.add_plot(
            y,y_hat,
            equal_aspect=True,
            label = "{}-CN-group(s) PAHs".format(no_sub)
        )
        
        if i == 1:
            plot.ax.legend(
                    prop = font_legend,loc = "center left",
                    bbox_to_anchor=(1.04,0.5), borderaxespad=0)
        
        plot.add_plot(
            limit[i],limit[i],
            scatter = False,
            plot_line=True, weight = (0,1),
            line_color = "black",
            xlim = tuple(limit[i]),
            ylim = tuple(limit[i]),
            xlabel = "Calculated " + elec_prop + " (eV)",
            ylabel = "QSPR prediction (eV)"
        )
        
        if no_sub == 4:
            plot.add_text(
                0.95,0.05,
                "{}\nR$^2$ = {:.2f}\nRMSD = {:.2f}eV".format(
                    label[i],glob_pearson,glob_rmsd)
            )
        plot.save_fig("test_"+elec_prop+".jpeg")

# %%
for i,p in enumerate(model_mt.parameters_list):
    print(p,model_mt.parameter_values[i])
    
for i,w in enumerate(model_mt.outputs):
    print(w.weight)

# %%
"""
plt.hist(total.loc[:,"no_subs"],bins=[1,1.5,2,2.5,3,3.5,4,4.5,5,5.5])
plt.savefig("subs_dist.jpeg",dpi=600)

plt.clf()

plt.hist(total.loc[:,"no_ring"],
         bins=[3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10])

plt.savefig("ring_dist.jpeg",dpi=600)
"""
# %%
np.random.seed(529)
random_seeds = np.random.randint(0,10e6,10)

overall_error = []

#avg_param = np.zeros(len(plist))
all_params = []
avg_weight = np.zeros((3,2))
avg_bias = np.zeros((3,1))

N = 10

for i in range(N):
    train = sampling(
        round(0.6*len(total)),
        total, 
        intervals = [1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0],
        random_state=random_seeds[i]
        )

    test = total.drop(index = train.index,axis=0)
    
    DPO = train.loc[:,dpo_label]

    SDPO = train.loc[:,sdpo_label]

    Y1 = np.array(train.loc[:,"BG"])
    Y2 = np.array(train.loc[:,"EA"])
    Y3 = np.array(train.loc[:,"IP"])
    
    model_mt = modGD_DPO(
        parameters_list=plist,
        parameter_values=list(np.random.rand(len(plist))/10),
        constant_list= clist,
        constant_values= cvalues,
        tasks = 3
    )
    
    model_mt = train_model(
        model_mt,
        DPO,SDPO,Y1,Y2,Y3,
        50,2)

    dpo = test.loc[:,"DPO"]

    sdpo = test.loc[:,sdpo_label]

    error = []
    for i,elec_prop in enumerate(["BG","EA","IP"]):
        for no_sub in range(1,6):
            if no_sub == 5:
                dpo = test.loc[:,"DPO"]
                sdpo = test.loc[:,sdpo_label]
                y = np.array(test.loc[:,elec_prop]).reshape(-1)

            else:
                dpo = test.loc[test.no_subs == no_sub].loc[:,"DPO"]
                sdpo = test.loc[test.no_subs == no_sub].loc[:,sdpo_label]
                y = np.array(
                    test.loc[test.no_subs == no_sub].loc[:,elec_prop]).reshape(-1)
                
            xmt1,xmt2,y_hat = model_mt.predict(dpo,sdpo,i)
            y_hat = y_hat.reshape(-1)

            error.append(RMSD(y,y_hat))
            
    overall_error.append(error)
    
    all_params.append(model_mt.parameter_values) 
    
    avg_bias += np.array(list(map(
        lambda x: x.weight[0], model_mt.outputs)))
    
    avg_weight += np.array(list(map(
        lambda x: x.weight[1], 
        model_mt.outputs))).squeeze(1)
    
avg_param = np.mean(all_params,axis = 0)
std_param = np.std(all_params,axis = 0)

avg_weight /= N
avg_bias /= N
std_overall_error = np.std(overall_error,0)
overall_error = np.mean(overall_error,0)

#%%
error = overall_error.reshape(3,5)
std = std_overall_error.reshape(3,5)
for i,elec_prop in enumerate(["BG","EA","IP"]):
    print("=============")
    for no_sub in range(1,6):
        if no_sub == 6:
            no_sub = "total"
        print("Error for {} of {}-times subtituted PAHs: {:.2f} +/- {:.2f} eV".format(
            elec_prop, no_sub, error[i][no_sub-1],std[i][no_sub-1]
        ))
    
print("parameter:")
for i,param in enumerate(model_mt.parameters_list):
    print(param + " = {:.2f} +/- {:.2f}".format(avg_param[i],std_param[i]))
print("weight",avg_weight)
print("bias",avg_bias)
#%%
"""
=============
Error for BG of 1-times subtituted PAHs: 0.11 +/- 0.00 eV
Error for BG of 2-times subtituted PAHs: 0.11 +/- 0.01 eV
Error for BG of 3-times subtituted PAHs: 0.12 +/- 0.01 eV
Error for BG of 4-times subtituted PAHs: 0.14 +/- 0.02 eV
Error for BG of 5-times subtituted PAHs: 0.12 +/- 0.00 eV
=============
Error for EA of 1-times subtituted PAHs: 0.08 +/- 0.00 eV
Error for EA of 2-times subtituted PAHs: 0.10 +/- 0.01 eV
Error for EA of 3-times subtituted PAHs: 0.12 +/- 0.01 eV
Error for EA of 4-times subtituted PAHs: 0.15 +/- 0.02 eV
Error for EA of 5-times subtituted PAHs: 0.10 +/- 0.00 eV
=============
Error for IP of 1-times subtituted PAHs: 0.09 +/- 0.00 eV
Error for IP of 2-times subtituted PAHs: 0.09 +/- 0.01 eV
Error for IP of 3-times subtituted PAHs: 0.13 +/- 0.01 eV
Error for IP of 4-times subtituted PAHs: 0.14 +/- 0.01 eV
Error for IP of 5-times subtituted PAHs: 0.10 +/- 0.00 eV
parameter:
a = 0.03 +/- 0.00
b = -0.19 +/- 0.01
c = 0.41 +/- 0.02
d = 0.36 +/- 0.01
s = 0.68 +/- 0.01
weight [[-0.58675826 -0.08852974]
 [ 0.23626365  0.42126777]
 [-0.34400856  0.33456266]]
bias [[4.57047674]
 [1.65906827]
 [6.20932295]]

"""
# %%
for i,param in enumerate(model_mt.parameters_list):
    print(param,model_mt.parameter_values[i])
# %%
"""x
=============
Error for BG of 1-times subtituted PAHs: 0.13 +/- 0.02 eV
Error for BG of 2-times subtituted PAHs: 0.15 +/- 0.04 eV
Error for BG of 3-times subtituted PAHs: 0.16 +/- 0.04 eV
Error for BG of 4-times subtituted PAHs: 0.17 +/- 0.04 eV
Error for BG of 5-times subtituted PAHs: 0.14 +/- 0.02 eV
=============
Error for EA of 1-times subtituted PAHs: 0.10 +/- 0.00 eV
Error for EA of 2-times subtituted PAHs: 0.18 +/- 0.02 eV
Error for EA of 3-times subtituted PAHs: 0.20 +/- 0.02 eV
Error for EA of 4-times subtituted PAHs: 0.27 +/- 0.03 eV
Error for EA of 5-times subtituted PAHs: 0.15 +/- 0.01 eV
=============
Error for IP of 1-times subtituted PAHs: 0.11 +/- 0.01 eV
Error for IP of 2-times subtituted PAHs: 0.16 +/- 0.02 eV
Error for IP of 3-times subtituted PAHs: 0.21 +/- 0.02 eV
Error for IP of 4-times subtituted PAHs: 0.23 +/- 0.03 eV
Error for IP of 5-times subtituted PAHs: 0.14 +/- 0.00 eV
parameter:
a = 0.05 +/- 0.01
b = -0.18 +/- 0.01
c = 0.40 +/- 0.02
d = 0.36 +/- 0.01
s = 0.19 +/- 0.03
weight [[-0.56264909 -0.08589838]
 [ 0.22917963  0.3975809 ]
 [-0.36585998  0.31165485]]
bias [[4.49775637]
 [1.83542882]
 [6.37953374]]
"""