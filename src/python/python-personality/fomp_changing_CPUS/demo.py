import models

import algos as alg
import math as mt
import numpy as np
import pandas as pd
import time as tm
import sys
    

def run_experiment(features, target, model, k_range, eps_FAST_OMP, tau, eps_DASH, alpha, r, N_samples, N_CPU, SDS_OMP = True, FAST_OMP = True, SDS_MA = True, DASH = True, Top_k = True) :

    '''
    Run a set of experiments for selected algorithms. All results are saved to text files.
    
    INPUTS:
    
    features -- the feature matrix
    target -- the observations
    model -- choose if the regression is linear or logistic
    k_range -- range for the solution size to test experiments with
    
    eps_FAST_OMP -- parameter epsilon for FAST_OMP
    tau -- parameter m/M for the FAST_OMP
    
    eps_DASH -- parameter epsilon for DASH
    alpha -- parameter (m/M)^2 for the DASH
    r -- number of outer iterations for DASH
    
    N_samples -- number of runs for the randomized algorithms
    SDS_OMP -- if True the SDS_OMP algorithm is tested
    FAST_OMP -- if True the FAST_OMP algorithm is tested
    SDS_MA -- if True the SDS_MA algorithm is tested
    DASH -- if True the DASH algorithm is tested
    Top_k -- if True the Top_k algorithm is tested
    '''
    
    length_data =  len(k_range) * len(N_CPU)
        
    # ----- run FAST_OMP
    
    if SDS_OMP :
        print('----- testing SDS_OMP')
        results = pd.DataFrame(data = {'k': np.zeros(length_data).astype('int'), 'n_cpus': np.zeros(length_data).astype('int'), 'time_mn': np.zeros(length_data), 'rounds_mn': np.zeros(length_data),'metric_mn': np.zeros(length_data), 'time_sd': np.zeros(length_data), 'rounds_sd': np.zeros(length_data),'metric_sd': np.zeros(length_data)})

        h = 0
        for j in range(len(k_range)) :

            for l in range(len(N_CPU)) :
        
                # perform experiments
                out = [alg.SDS_OMP(features, target, model, k_range[j], N_CPU[l]) for i in range(N_samples)]
                out = np.array(out)
            
                # save data to file
                results.loc[j + h + l,'k']         = k_range[j]
                results.loc[j + h + l,'n_cpus']    = N_CPU[l]
                results.loc[j + h + l,'time_mn']   = np.mean([out[i,0] for i in range(N_samples)])
                results.loc[j + h + l,'rounds_mn'] = np.mean([out[i,1] for i in range(N_samples)])
                results.loc[j + h + l,'metric_mn'] = np.mean([out[i,2] for i in range(N_samples)])
                results.loc[j + h + l,'time_sd']   = np.std([out[i,0]  for i in range(N_samples)])
                results.loc[j + h + l,'rounds_sd'] = np.std([out[i,1]  for i in range(N_samples)])
                results.loc[j + h + l,'metric_sd'] = np.std([out[i,2]  for i in range(N_samples)])
                results.to_csv('SDS_OMP.csv', index = False)
        
            h += l
        
        
       
        
    # ----- run Top_k
    
    if Top_k :
    
        print('----- testing Top_k')
        results = pd.DataFrame(data = {'k': np.zeros(length_data).astype('int'), 'n_cpus': np.zeros(length_data).astype('int'), 'time': np.zeros(length_data), 'rounds': np.zeros(length_data),'rounds_ind': np.zeros(length_data),'metric': np.zeros(length_data)})
                
        h = 0
        for j in range(len(k_range)) :

            for l in range(len(N_CPU)) :
        
                # perform experiments
                out = alg.Top_k(features, target, k_range[j], model, N_CPU[l])
                out = np.array(out)
            
                # save data to file
                results.loc[j + h + l,'k'] = k_range[j]
                results.loc[j + h + l,'n_cpus'] = N_CPU[l]
                results.loc[j + h + l,'time']   = out[0]
                results.loc[j + h + l,'rounds'] = out[1]
                results.loc[j + h + l,'rounds_ind'] = out[2]
                results.loc[j + h + l,'metric'] = out[3]
                results.to_csv('Top_k.csv', index = False)

            h +=l
        
        
    # ----- run FAST_OMP
    if FAST_OMP :
        print('----- testing FAST_OMP')
        results = pd.DataFrame(data = {'k': np.zeros(length_data).astype('int'), 'n_cpu': np.zeros(length_data).astype('int'), 'time_mn': np.zeros(length_data), 'rounds_mn': np.zeros(length_data),'metric_mn': np.zeros(length_data), 'time_sd': np.zeros(length_data), 'rounds_sd': np.zeros(length_data),'metric_sd': np.zeros(length_data)})
        
        h = 0
        for j in range(len(k_range)) :
        
            for l in range(len(N_CPU)) :
        
                # perform experiments
                out = [alg.FAST_OMP(features, target, model, k_range[j], eps_FAST_OMP, tau, N_CPU[l]) for i in range(N_samples)]
                out = np.array(out)
            
                # save data to file
                results.loc[j + h+l,'k']         = k_range[j]
                results.loc[j + h+l,'n_cpu']     = N_CPU[l]
                results.loc[j + h+l,'time_mn']   = np.mean([out[i,0] for i in range(N_samples)])
                results.loc[j + h+l,'rounds_mn'] = np.mean([out[i,1] for i in range(N_samples)])
                results.loc[j + h+l,'rounds_ind_mn'] = np.mean([out[i,2] for i in range(N_samples)])
                results.loc[j + h+l,'metric_mn'] = np.mean([out[i,3] for i in range(N_samples)])
                results.loc[j + h+l,'time_sd']   = np.std([out[i,0]  for i in range(N_samples)])
                results.loc[j + h+l,'rounds_sd'] = np.std([out[i,1]  for i in range(N_samples)])
                results.loc[j + h+l,'rounds_ind_sd'] = np.std([out[i,2] for i in range(N_samples)])
                results.loc[j + h+l,'metric_sd'] = np.std([out[i,3]  for i in range(N_samples)])
                results.to_csv('FAST_OMP.csv', index = False)
                
            h += l
    
'''
Test algorithms with the run_experiment function.
target -- the observations for the regression
features -- the feature matrix for the regression
model -- choose if 'logistic' or 'linear' regression
k_range -- range for the parameter k for a set of experiments
SDS_OMP -- if True, test this algorithm
FAST_OMP -- if True, test this algorithm
SDS_MA -- if True, test this algorithm
DASH -- if True, test this algorithm
Top_k -- if True, test this algorithm
eps_FAST_OMP -- parameter epsilon for FAST_OMP
eps_DASH -- parameter epsilon for DASH
tau -- parameter m/M for the FAST_OMP
alpha -- parameter (m/M)^2 for the FAST_OMP
r -- number of outer iterations for DASH
N_samples -- number of runs for the randomized algorithms
This set of experiment was tested on Python 3.6.5, with MacBook Pro with processor 2,7 GHz Dual-Core Intel Core i5 and 8 GB 1867 MHz DDR3 memory. The parameters for all algorithms may not be optimally tuned. The performance of each algorithm is affected by the machine used to run the algorithms.
'''

# define features and target for the experiments
#df = pd.read_csv('healthstudy.csv', index_col=0, parse_dates=False)
df = pd.read_csv('healthstudy.csv', index_col=0, parse_dates=False)
df = pd.DataFrame(df)

#frames = [df, df, df, df, df, df, df, df, df, df,df, df, df, df, df, df, df, df, df, df]
#frames = [df, df, df, df, df, df, df, df, df, df,df, df, df, df, df, df, df, df, df, df]
#df = pd.concat(frames, sort=False)

target = df.iloc[:,0]
features = df.iloc[:, range(1, df.shape[1])]
print(sys.getsizeof(target)/1024/1024)
print(sys.getsizeof(features)/1024/1024)
print (features.shape)
# choose if logistic or linear regression
model = 'logistic'

# set range for the experiments
k_range = np.array([1, 1000, 2000])
N_CPUS = np.array(range(1, 40))
N_CPUS = np.array([1, 2, 4, 8, 16, 20, 24, 28, 32, 36, 40])

# choose algorithms to be tested
SDS_MA   = False
SDS_OMP  = True 
FAST_OMP = True
Top_k    = True 
DASH     = False
# define parameters for algorithms

eps_FAST_OMP = 0.999
eps_DASH = 0.000000001

tau = 0.1
alpha = tau * tau
r = mt.ceil(20/eps_DASH * np.log(features.shape[1])/ np.log(1 + eps_DASH/2))

# number of samples per evaluation
N_samples = 3 

# run experiment
run_experiment(features, target, model = model, k_range = k_range, eps_FAST_OMP = eps_FAST_OMP, tau = tau,eps_DASH = eps_DASH, alpha = alpha, r = r, N_samples = N_samples, N_CPU = N_CPUS, SDS_OMP = SDS_OMP, SDS_MA = SDS_MA, DASH = DASH, FAST_OMP = FAST_OMP, Top_k = Top_k)
