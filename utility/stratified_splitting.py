import numpy as np
import pandas as pd

import sys
import os
module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])
sys.path.append(module_root)



def random_sampling(sample_size,data,random_state):
    train_set = data.sample(n = sample_size, random_state=random_state)
    return train_set

def stratified_sampling(sample_size,data,intervals,random_state):
    """
    Stratified sampling a data. Return a sub-dataset with specified 
    size. The subdata has the distribution of BG imitates that of the 
    original data.
    Args:
        sample_size (int): the size of the subset
        data (pd.DataFrame): the data set from which subset is drawn.
        intervals (list): 
        random_state (int): seed
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
            ].sample(n=round(intervalSize[i]*(sample_size/len(data))),random_state = intervalSeeds[i])
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
    if intervals:
        return stratified_sampling(sample_size,data,intervals,random_state)
    else:
        return random_sampling(sample_size,data,random_state)

