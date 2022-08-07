#%%
import numpy as np
from gd_dpo.mod_GDDPO import modGD_DPO
from utility.stratified_splitting import sampling

def pearson_corr(X,Y):
    X = np.array(X).reshape(-1)
    Y = np.array(Y).reshape(-1)
    X_m = np.mean(X)*np.ones(len(X)) - X
    Y_m = np.mean(Y)*np.ones(len(Y)) - Y
    
    R = np.sqrt(np.sum(X_m**2))*np.sqrt(np.sum(Y_m**2))
    R = np.sum(np.dot(X_m.T,Y_m)) / R
    
    return R

def RMSD(Y_hat,Y):
    Y_hat = np.array(Y_hat).reshape(-1)
    Y = np.array(Y).reshape(-1)

    rmsd = (Y_hat - Y)**2
    rmsd = np.mean(rmsd)
    rmsd = np.sqrt(rmsd)
    return rmsd

def train_model(
    model_mt,DPO,SDPO,
    Y1,Y2,Y3,epochs,lr,
    return_loss = False
    ):
    if return_loss:
        loss_collection= []
        
    for i in range(epochs):
        if  i%3 == 0:
            loss, dL1 = model_mt.feedforward(DPO,SDPO,Y1,0)
            if loss > 0.009:
                model_mt.step(lr,dL1)
    
        elif i%3 == 1:
            loss, dL2 = model_mt.feedforward(DPO,SDPO,Y2,1)
            if loss > 0.009:
                model_mt.step(lr,dL2)
        elif i%3 == 2:
            loss, dL3 = model_mt.feedforward(DPO,SDPO,Y3,2)
            if loss > 0.009:
                model_mt.step(lr,dL3)
    
        _, dL2 = model_mt.feedforward(DPO,SDPO,Y2,1)
        _, dL3 = model_mt.feedforward(DPO,SDPO,Y3,2)
        """
        print("Loss:BG = {:.4f}, EA = {:.4f}, IP = {:.4f}".format(
            np.sqrt(loss1),np.sqrt(loss2),np.sqrt(loss3)
        ))
        """
        
        if return_loss:
            loss_collection.append([loss])     
    
    if return_loss:
        return model_mt, loss_collection
    else:
        return model_mt

def repeat_train_model(
    N, total, seed,
    dpo_label = "Trun_DPO", sdpo_label = "SDPO"
    ):
    
    np.random.seed(seed)
    random_seeds = np.random.randint(0,10e6,10)

    overall_error = []

    avg_param = np.zeros(5)
    avg_weight = np.zeros((3,2))
    avg_bias = np.zeros((3,1))

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
            parameters_list=["s","a","b","c","d"],
            parameter_values=list(np.random.rand(5)/10),
            constant_list= ["sa","b0","bf","cf"],
            constant_values= [1,0,0,0],
            tasks = 3
        )
        
        model_mt = train_model(
            model_mt,
            DPO,SDPO,Y1,Y2,Y3,
            50,2)

        dpo = test.loc[:,"Trun_DPO"]

        sdpo = test.loc[:,sdpo_label]

        error = []
        for i,elec_prop in enumerate(["BG","EA","IP"]):
            dpo = test.loc[:,"Trun_DPO"]
            sdpo = test.loc[:,sdpo_label]
            y = np.array(test.loc[:,elec_prop]).reshape(-1)

            _,_,y_hat = model_mt.predict(dpo,sdpo,i)
            y_hat = y_hat.reshape(-1)

            error.append(RMSD(y,y_hat))
                
        print("epoch: ",i,"error: ",error)
        overall_error.append(error)
            
        avg_param += model_mt.parameter_values
        
        avg_bias += np.array(list(map(
            lambda x: x.weight[0], model_mt.outputs)))
        
        avg_weight += np.array(list(map(
            lambda x: x.weight[1], 
            model_mt.outputs))).squeeze(1)
    
    overall_error = np.mean(overall_error,0)
    print("Finished, error: ",overall_error)
            
    avg_param /= N
    avg_weight /= N
    avg_bias /= N
    
    avg_model = modGD_DPO(
        parameters_list=["s","a","b","c","d"],
        parameter_values=list(avg_param),
        constant_list= ["sa","b0","bf","cf"],
        constant_values= [1,0,0,0],
        tasks = 3
    )
    
    for i,b in enumerate(avg_bias):
        avg_model.outputs[i].weight = (
            b, avg_weight[i][np.newaxis]
        )
    
    return avg_model


# %%
"""
import pandas as pd
import matplotlib.pyplot as plt

total = pd.read_csv(
    "total_data.csv"
    #"total_data_plus_pah_n_thieno.csv"
    )
total = total.fillna(0)

model = repeat_train_model(
    2,total,10
)
# %%
error_list = []
for n in range(900):
    dpo = total.loc[n,"Trun_DPO"]
    sdpo = total.loc[n,"SDPO"]
    y = total.loc[n,"EA"]

    x1,x2,y_hat = model.predict([dpo],[sdpo],task=1)

    error_list.append(abs(y-y_hat[0][0]))

# %%
plt.hist(
    error_list, 
    bins = np.arange(0,0.30,0.05)
)
# %%
"""