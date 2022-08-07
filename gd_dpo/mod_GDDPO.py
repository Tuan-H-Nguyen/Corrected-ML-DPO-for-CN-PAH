#%% import stuffs
import sys
import os
module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-2])
sys.path.append(module_root)
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym #PVP
from sklearn.linear_model import LinearRegression
from copy import deepcopy

import time

def compute_r2(x,y):
    sum_xy = np.sum(x*y)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x**2)
    sum_y2 = np.sum(y**2)
    n = len(x)
    r = (n*sum_xy - sum_x*sum_y)/(((n*sum_x2-sum_x**2)*(n*sum_y2-sum_y**2))**0.5)
    return r**2

class DpoEvaluator:
    def __init__(
        self,
        parameters_list,
        parameter_values,
        constant_list,
        constant_values
        ):
        self.parameters_list = [
            sym.symbols(p) for p in parameters_list]
        self.parameter_values = parameter_values
        self.constant_list = [
            sym.symbols(p) for p in constant_list]
        self.constant_values = constant_values
        self.exp_dict = {}

    def update_pvalues(self,parameter_values):
        self.parameter_values = parameter_values

    def eval(self,exp):
        try: exp_f = self.exp_dict[exp]
        except KeyError:
            exp_f = sym.lambdify(
                self.parameters_list + self.constant_list,exp)
            self.exp_dict[exp] = exp_f
        return exp_f(*np.hstack([self.parameter_values,self.constant_values]))

class DpoDerivative:
    def __init__(
        self,
        parameters_list,
        parameter_values,
        constant_list,
        constant_values):
        self.parameters_list = [
            sym.symbols(p) for p in parameters_list]
        self.parameter_values = parameter_values
        self.constant_list = [
            sym.symbols(p) for p in constant_list]
        self.constant_values = constant_values
        self.d_dict = {}
    
    def differentiate(self,exp):
        return [sym.diff(exp,p) for p in self.parameters_list]

    def grad(self,exp):
        try:
            d_exp = self.d_dict[exp]
        except KeyError:
            d_exp = self.differentiate(exp)
            d_exp = list(map(
                lambda f: sym.lambdify(
                    self.parameters_list + self.constant_list,f
                    ),d_exp
                ))
            self.d_dict[exp] = d_exp
        d_exp = np.array(
            list(map(
                lambda f:f(*np.hstack([self.parameter_values,self.constant_values])),d_exp
            )))
        return d_exp

    def update_parameter(self,parameter_values):
        self.parameter_values = parameter_values

class Loss:
    def __init__(self):
        loss_record = []
    def dL(self):
        return (2/len(self.dY))*self.dY
    def eval(self,Y_hat,Y):
        dY = Y_hat - Y
        self.dY = dY
        return (1/len(dY))*np.sum(dY**2)
    
class LinearOutput:
    def __init__(self,weight=None):
        self.weight = weight 
        
    def least_square(self,X,Y):
        lm = LinearRegression().fit(X,Y.reshape(-1,1))
        wb, w = lm.intercept_, lm.coef_
        self.weight = wb,w
        
    def forward(self,X):
        wb,w = self.weight
        return np.dot(w,X.T) + wb
    
    def dL(self):
        return self.weight[1]
        
class modGD_DPO:
    def __init__(
        self,
        parameters_list,
        parameter_values = None,
        constant_list = [],
        constant_values = None,
        weight = None,
        tasks = 1
        ):
        """
        Args:
            parameters_list:
        """
        if not parameter_values:
            #parameter_values = np.random.randn(len(parameters_list))*0.1
            parameter_values = np.zeros(len(parameters_list))
        if not constant_values:
            constant_values = np.zeros(len(constant_list))
        self.parameters_list = parameters_list
        self.parameter_values = parameter_values
        self.constant_list = constant_list
        self.constant_values = np.array(constant_values)
        if not weight:
            weight = (0,0)
        self.weight = weight
        self.E = DpoEvaluator(
            self.parameters_list,self.parameter_values,
            self.constant_list, self.constant_values
            )
        self.D = DpoDerivative(
            self.parameters_list,self.parameter_values,
            self.constant_list,self.constant_values
            )
        self.L = Loss()
        
        self.outputs = [
            LinearOutput() for _ in range(tasks)] 
    
    def feedforward(
        self,
        DPO,
        SDPO,
        Y,
        task
        ):
        """
        Args:
            DPO (list or np.array): array of DPO expressions
            Y (np.array): array of target
            training (bool): True if training, that is, the weight and bias 
                of linear line is readily computed and only return the grad
                -ient of loss respect to DPO parameters.
                False to compute the current model loss.
            prediction_mode (bool): True to return the DPO values and 
                predicted targets.
        """
        X1 = np.array(list(map(self.E.eval,DPO))).reshape(-1,1)
        X2 = np.array(list(map(self.E.eval,SDPO))).reshape(-1,1)
        X = np.hstack([X1,X2])

        self.outputs[task].least_square(X , Y)
        
        Y_hat = self.outputs[task].forward(X)
        loss = self.L.eval(Y_hat.reshape(-1),Y)
        
        dX1 = np.array(list(map(self.D.grad,DPO)))[:,:,np.newaxis]
        dX2 = np.array(list(map(self.D.grad,SDPO)))[:,:,np.newaxis]
        
        dX = np.concatenate([dX1,dX2],axis = -1)
        #print(np.any(dX))
        dX = np.dot(dX,self.outputs[task].dL().T)
        #print(np.any(dX))
        dX = dX.squeeze(-1)
        
        dL_p = np.dot(dX.T,self.L.dL().T)
        dL_p = dL_p.reshape(-1)
        return loss,dL_p
    
    def compute_loss(self,DPO,SDPO,Y,task):
        _,_,Y_hat = self.predict(DPO,SDPO,task)
        loss = self.L.eval(Y_hat,Y)
        return loss
    
    def predict(self,DPO,SDPO,task):
        X1 = np.array(list(map(self.E.eval,DPO))).reshape(-1,1)
        X2 = np.array(list(map(self.E.eval,SDPO))).reshape(-1,1)
        X = np.hstack([X1,X2])
        
        Y_hat = self.outputs[task].forward(X)
        return X1,X2,Y_hat

    def step(self,lr,dL_p):
        self.parameter_values -= lr*dL_p
        self.E.update_pvalues(self.parameter_values)
        self.D.update_parameter(self.parameter_values)

    def rewind(self,backUpParam,backUpWeight):
        self.parameter_values = backUpParam
        self.weight = backUpWeight
        self.E.update_pvalues(self.parameter_values)
        self.D.update_parameter(self.parameter_values)
    
    def fit(
        self,
        DPO,Y,
        lr,
        epochs,lr_decay_rate=10, min_lr = 10**-9,
        threshold= 10**-9, patience = 50,
        verbose = 1, 
        validation_set = None,
        record = False,fixed_weight =False):
        count = 0
        valL = 0
        if record:
            self.train_loss = []
            self.val_loss = []
        for epoch in range(1,epochs+1):
            #back up parameters
            backUpParam = deepcopy(self.parameter_values)
            backUpWeight = deepcopy(self.weight)
            #feedforward to compute loss and gradient of L respect to DPO params
            loss,grad = self.feedforward(DPO,Y,fixed_weight)
            #update base on gradient
            self.step(lr,grad)
            # compute loss post-update
            updatedLoss = self.compute_loss(DPO,Y)
            if updatedLoss > loss:
                print("Loss increase. Rewind parameters and decay learning rate.")
                self.rewind(backUpParam,backUpWeight)
                lr /= lr_decay_rate
                if lr <= min_lr:
                    print("Break due to small learning rate.")
                    break
            elif loss - updatedLoss < threshold:
                count += 1
                if count > patience:
                    print("Break due to insignificant improvement.")
                    break
            else:
                count = 0
            if validation_set:
                valL = self.compute_loss(*validation_set)
            if verbose != 0 and epoch%verbose == 0:
                print("Epoch {}: loss: {}, validation loss: {}".format(epoch,updatedLoss,valL))
                print(self.parameter_values)
                print(self.weight)
            if record:
                self.train_loss.append(updatedLoss)
                self.val_loss.append(valL)
            
        self.finalEpoch = epoch
        self.finalTrainLoss = updatedLoss
        self.finalValLoss = valL
        
#%%
