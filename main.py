import argparse

import numpy as np
import pandas as pd

from gd_dpo.mod_GDDPO import modGD_DPO
from trainer import repeat_train_model, train_model

from MolGraph.poly_rings.DPO import DPO_generate
from substituent import AugDPO

parser = argparse.ArgumentParser(
    description="""For either training the model or predicting using the pre-trained model.
    If is advised to train the model, save trained model' parameters to a checkpoint file (
    path provided with -chk argument) by using -t argument. 
    Afterward, prediction can be made freely from retrieved the model by porviding
    the SMILES string and the path to checkpoint file."""
)
parser.add_argument(
    "smiles",
    help = """Smiles string for PAH or thienoacenes. 
    To make prediction, users only need to provide this argument. 
    Note that this argument is irrelevant if argument -t is used, 
    therefore just provide 0 if train the model."""
)
parser.add_argument(
    "-t","--train",
    help = "Training the model and save model to checkpoint file. Specify this will train model instead of predicting.",
    default = 0
)
parser.add_argument(
    "-chk","--checkpoint",
    help = "Path to file that is used to save (training model) or to retrieve model's parameters (predicting). Default to checkpoint.txt.",
    default = "checkpoint.txt"
)
parser.add_argument(
    "-n","--num",
    help = "Number of runs to perform training testing and return average model. Irrelevant if predicting.",
    type = int,
    default = 1
)
parser.add_argument(
    "-d","--data",
    help = "Total data to sample for training and testing data set. Irrelevant if predicting.",
    default = "total_data.csv"
)
parser.add_argument(
    "-s","--seed",
    help = "Random seed. Irrelevant if predicting.",
    default = 2022,
    type = int
)

args = parser.parse_args()

def str2list(
    line,
    to_float = False,
    ):
    _list = line.split(";")
    if to_float:
        _list = [float(i) for i in _list]
    return _list

def retrieve_model(file):
    with open(file, "r") as o:
        line = o.readline()
        p_list = str2list(line,to_float = False)
        line = o.readline()
        pv_list = str2list(line,to_float = True)
        line = o.readline()
        c_list = str2list(line,to_float = False)
        line = o.readline()
        cv_list = str2list(line,to_float = True)
        line = o.readline()
        task = int(line)
        
        model = modGD_DPO(
            parameters_list=p_list,
            parameter_values=pv_list,
            constant_list=c_list,
            constant_values=cv_list,
            tasks = task
        )
        
        for i in range(task):
            line = o.readline()
            line = str2list(line,to_float = True)
            
            bias = np.array([line[0]])
            weight = np.array(line[1:])[np.newaxis]
            
            model.outputs[i].weight = (bias,weight)
        
    return model
        
def list2str(_list):
    """Turning list to writable string

    Args:
        _list (list): 

    Returns:
        string: 
    """
    result = [str(item) for item in _list]
    return ";".join(result)

def save_model(model,path):
    """Write model's parameters to external text file

    Args:
        model (GD_DPO model): 
        path (string): path for writing
    """
    with open(path,"w") as o:
        o.write(list2str(model.parameters_list))
        o.write("\n")
        o.write(list2str(model.parameter_values))
        o.write("\n")
        o.write(list2str(model.constant_list))
        o.write("\n")
        o.write(list2str(model.constant_values))
        o.write("\n")
        o.write(str(len(model.outputs)))
        o.write("\n")
        
        for i in range(len(model.outputs)):
            foo = list(model.outputs[i].weight)
            foo[1] = foo[1].squeeze()
            foo = list(np.hstack(foo))
            
            o.write(list2str(foo))
            o.write("\n")

if int(args.train):
    data = pd.read_csv(args.data)
    # to simplify the argument need to be provided for training, default 
    # training hyperparameter and arguments specified in the Trainer 
    # object are used
    model = repeat_train_model(
        N = args.num,
        total = data,
        seed = args.seed
        )
    save_model(model,args.checkpoint)

else:
    model = retrieve_model(args.checkpoint)
    dpo = [DPO_generate(args.smiles)]
    
    aug_dpo = AugDPO(overlayer_effect=False)
    sdpo = [aug_dpo.map(args.smiles)]
    
    print("### PREDICTION ###")
    
    _,_, y = model.predict(dpo,sdpo, task = 0)
    print("Band gap: {:.2f}eV".format(y[0][0]))
    
    _,_, y = model.predict(dpo,sdpo, task = 1)
    print("Electron Affinity: {:.2f}eV".format(y[0][0]))
    
    _,_, y = model.predict(dpo,sdpo, task = 2)
    print("Ionization Potential: {:.2f}eV".format(y[0][0]))
