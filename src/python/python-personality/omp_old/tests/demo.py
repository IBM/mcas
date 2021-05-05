import pymcas
import omp.models
import omp.algos as alg
import os
import sys
import math as mt
import numpy as np
import pandas as pd
import time as tm
from sys import getsizeof
import cProfile, pstats, io

def ado_run_experiment (features, params):
   target = ado.load('target')
   model = params['model']
   selected_size = params['selected_size'] 
   alg_type = params['alg_type'] 
   from omp import algos as alg
   if (alg_type == "SDS_OMP"):
      out = alg.SDS_OMP(features, target, model, selected_size)
   return out 

     # set range for the experiments

#   target_df = target_df + 1
    # this is sent back to client as invoke result


def run_experiment(features, target, model, k_range, SDS_OMP = True, SDS_MA = True, Top_k = True) :

    '''
    Run a set of experiments for selected algorithms. All results are saved to text files.
    
    INPUTS:
    
    features -- the feature matrix
    target -- the observations
    model -- choose if the regression is linear or logistic
    k_range -- range for the solution size to test experiments with

    SDS_OMP -- if True the SDS_OMP algorithm is tested
    SDS_MA -- if True the SDS_MA algorithm is tested
    Top_k -- if True the Top_k algorithm is tested

    '''
    # ----- run SDS_OMP
    if SDS_OMP :
        print('----- testing SDS_OMP')
        results = pd.DataFrame(data = {'k': np.zeros(len(k_range)).astype('int'), 'time': np.zeros(len(k_range)), 'rounds': np.zeros(len(k_range)),'metric': np.zeros(len(k_range))})
        for j in range(len(k_range)) :
            
            # parameters dor the experiments
            params = {
                    'model' : model,
                    'selected_size' : k_range[j],
                    'alg_type' : "SDS_OMP"
                    }

            # perform experiments
#            out = alg.SDS_OMP(features, target, model, k_range[j])
            out = pool.invoke('features', ado_run_experiment, params) # the experiment run on the server

            print (out)
            exit(0)
            out = np.array(out)
            # save data to file
            results.loc[j,'k'] = k_range[j]
            results.loc[j,'time']   = out[0]
            results.loc[j,'rounds'] = out[1]
            results.loc[j,'metric'] = out[2]
            results.to_csv('SDS_OMP.csv', index = False)
        #res = alg.SDS_OMP(features, target, model, k_range[-1])
        
        
        
    # ----- run SDS_MA
    if SDS_MA :
        print('----- testing SDS_MA')
        results = pd.DataFrame(data = {'k': np.zeros(len(k_range)).astype('int'), 'time': np.zeros(len(k_range)), 'rounds': np.zeros(len(k_range)),'metric': np.zeros(len(k_range))})
        for j in range(len(k_range)) :
        
            # perform experiments
            out = alg.SDS_MA(features, target, model, k_range[j])
            out = np.array(out)
            
            # save data to file
            results.loc[j,'k'] = k_range[j]
            results.loc[j,'time']   = out[0]
            results.loc[j,'rounds'] = out[1]
            results.loc[j,'metric'] = out[2]
            results.to_csv('SDS_MA.csv', index = False)
        
        
        
    # ----- run Top_k
    if Top_k :
        print('----- testing Top_k')
        results = pd.DataFrame(data = {'k': np.zeros(len(k_range)).astype('int'), 'time': np.zeros(len(k_range)), 'rounds': np.zeros(len(k_range)),'metric': np.zeros(len(k_range))})
        for j in range(len(k_range)) :
        
            # perform experiments
            out = alg.Top_k(features, target, k_range[j], model)
            out = np.array(out)
            
            # save data to file
            results.loc[j,'k'] = k_range[j]
            results.loc[j,'time']   = out[0]
            results.loc[j,'rounds'] = out[1]
            results.loc[j,'metric'] = out[2]
            results.to_csv('Top_k.csv', index = False)
    
    return
    
    
'''

Test algorithms with the run_experiment function.

target -- the observations for the regression
features -- the feature matrix for the regression
model -- choose if 'logistic' or 'linear' regression

k_range -- range for the parameter k for a set of experiments

SDS_OMP -- if True, test this algorithm
SDS_MA -- if True, test this algorithm
Top_k -- if True, test this algorithm

'''
session = pymcas.create_session(os.getenv('SERVER_IP'), 11911, debug=3)
if sys.getrefcount(session) != 2:
    raise ValueError("session ref count should be 2")
pool = session.create_pool("myPool", 1024*1024*1024)
if sys.getrefcount(pool) != 2:
    raise ValueError("pool ref count should be 2")

# define features and target for the experiments
path = os.path.abspath(os.path.dirname(sys.argv[0]))
dataname = 'short_data.csv'
fullpath_dataname = os.path.join(path, dataname)
df = pd.read_csv(fullpath_dataname, index_col=0, parse_dates=False)
df = pd.DataFrame(df)
target = df.iloc[:, -1]
features = df.iloc[:, range(df.shape[1] - 1)]

pool.save('features', features);
pool.save('target', target)
# choose if logistic or linear regression
model = 'logistic'

# set range for the experiments
k_range = range(100, 101)

# choose algorithms to be tested
SDS_OMP  = True 
Top_k    = False 
SDS_MA   = False
# run experiment
run_experiment(features, target, model = model, k_range = k_range, SDS_OMP = SDS_OMP, SDS_MA = SDS_MA, Top_k = Top_k)
