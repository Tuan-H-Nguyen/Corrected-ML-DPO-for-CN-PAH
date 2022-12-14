#%%
import numpy as np
import pandas as pd

import sys
import os
module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])

#sys.path.append(module_root)
#sys.path.append("C:\\Users\\hoang\\Dropbox\\Coding\\Cheminformatic\\MolGraph")

from poly_rings.DPO import DPO_generate

def random_sampling(sample_size,data,random_state):
    """
    sampling a data. Return a sub-dataset of the given 
    with specified size. 

    Args:
    + sample_size (int): the size of the returned subset
    + data (pd.DataFrame): the data set from which subset is drawn.
    + random_state (int): seed
    Return:
    + pd.DataFrame
    """
    train_set = data.sample(n = sample_size, random_state=random_state)
    return train_set

def stratified_sampling(sample_size,data,intervals,random_state):
    """
    Stratified sampling a data. Return a sub-dataset of the given 
    with specified size. The subdata are generated by 1/ binning the
    data according to the given intervals, 2/ sampling from each bin 
    round{[sample_size/(#samples all data)]*(#samples in bin)} samples,
    3/ concatenate data sampled from all bins
    Args:
    + sample_size (int): the size of the returned subset
    + data (pd.DataFrame): the data set from which subset is drawn.
    + intervals (list of float): [...,v_(i-1), v_i, v_(i+1)...] where 
        [v_(i-1),v_i] is a bin, [v_i,v_(i+1)] is a bin, ...
    + random_state (int): seed
    Return:
    + pd.DataFrame
    """
    np.random.seed(random_state)
    intervalSeeds = np.random.randint(np.iinfo(np.int32).max,size = len(intervals))
    #determine the number of samples of each interval
    intervalSize = [
        len(data.loc[(data.BG>intervals[i]) & (data.BG<intervals[i+1])]) 
        for i in range(len(intervals)-1)]
    #for each interval, pick a number of sample equal number of 
    #samples in that interval scale with quotient of sample size and 
    # data size and put all into one data set
    sampling = pd.concat([data.loc[
            (data.BG>intervals[i]) & (data.BG<intervals[i+1])
            ].sample(
                n=round(intervalSize[i]*(sample_size/len(data))),
                random_state = intervalSeeds[i])
            for i in range(len(intervals)-1)])
    #determine gap of sampled set and desired data set
    add_on = sample_size - len(sampling)
    #if the set is lack, then sample randomly
    if add_on > 0:
        spare = data.drop(sampling.index,axis=0).sample(n=add_on)
        sampling = pd.concat([sampling,spare])
    elif add_on < 0:
        spare = sampling.sample(n = abs(add_on))
        sampling = sampling.drop(spare.index)
    return sampling

def sampling(sample_size,data,intervals,random_state):
    """
    Either randomly or stratifiedly sampling a dataset

    Args:
    + sample_size (int): the size of the returned subset
    + data (pd.DataFrame): the data set from which subset is drawn.
    + intervals (list): if None or False, random sampling. If a
        list with format as above, stratified sampling.
    + random_state (int): seed
    Return:
    + pd.DataFrame
    """
    if intervals:
        return stratified_sampling(sample_size,data,intervals,random_state)
    else:
        return random_sampling(sample_size,data,random_state)

def replaceTrunDPO(data,smiles_data):
    data = data.drop("DPO equation",axis=1)
    data.at[:,'DPO equation'] = smiles_data.loc[:,'smiles'].apply(
        DPO_generate)
    return data

# %%
