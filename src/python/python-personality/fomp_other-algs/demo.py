import models

import algos as alg
import math as mt
import numpy as np
import pandas as pd
import time as tm
import sys
import random

def run_experiment(model, k_range, eps_FAST_OMP, tau, eps_DASH, alpha, r, N_samples, SDS_OMP = True, FAST_OMP = True, SDS_MA = True, DASH = True, Top_k = True, Oblivious = True, Random = True) :

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
        
    # ----- run SDS_OMP
    #if SDS_OMP :
        #print('----- testing SDS_OMP')
        #results = pd.DataFrame(data = {'k': np.zeros(len(k_range)).astype('int'), 'time': np.zeros(len(k_range)), 'rounds': np.zeros(len(k_range)),'metric': np.zeros(len(k_range))})
        #for j in range(len(k_range)) :
        
            # perform experiments
            #out = alg.SDS_OMP(features, target, model, k_range[j])
            #out = np.array(out)
            
            # save data to file
            #results.loc[j,'k'] = k_range[j]
            #results.loc[j,'time']   = out[0]
            #results.loc[j,'rounds'] = out[1]
            #results.loc[j,'metric'] = out[2]
            #results.to_csv('SDS_OMP.csv', index = False)
    ##res = alg.SDS_OMP(features, target, model, k_range[-1])
        
    # ----- run FAST_OMP
    if SDS_OMP :
        print('----- testing SDS_OMP')
        #results = pd.DataFrame(data = {'k': np.zeros(len(k_range)).astype('int'), 'time_mn': np.zeros(len(k_range)), 'rounds_mn': np.zeros(len(k_range)),'metric_mn': np.zeros(len(k_range)), 'time_sd': np.zeros(len(k_range)), 'rounds_sd': np.zeros(len(k_range)),'metric_sd': np.zeros(len(k_range))})
        alg.SDS_OMP(model, k_range[-1])
        #for j in range(len(k_range)) :
        
            # perform experiments
        #    out = [alg.SDS_OMP(features, target, model, k_range[j]) for i in range(N_samples)]
        #    out = np.array(out)
            
            # save data to file
        #    results.loc[j,'k']         = k_range[j]
        #    results.loc[j,'time_mn']   = np.mean([out[i,0] for i in range(N_samples)])
        #    results.loc[j,'rounds_mn'] = np.mean([out[i,1] for i in range(N_samples)])
        #    results.loc[j,'metric_mn'] = np.mean([out[i,2] for i in range(N_samples)])
        #    results.loc[j,'time_sd']   = np.std([out[i,0]  for i in range(N_samples)])
        #    results.loc[j,'rounds_sd'] = np.std([out[i,1]  for i in range(N_samples)])
        #    results.loc[j,'metric_sd'] = np.std([out[i,2]  for i in range(N_samples)])
        #    results.to_csv('SDS_OMP.csv', index = False)

     
    # ----- run TopK
    if Top_k :
        print('----- testing Top_k')
        results = pd.DataFrame(data = {'k': np.zeros(len(k_range)).astype('int'), 'time_mn': np.zeros(len(k_range)), 'rounds_mn': np.zeros(len(k_range)),'metric_mn': np.zeros(len(k_range)), 'time_sd': np.zeros(len(k_range)), 'rounds_sd': np.zeros(len(k_range)),'metric_sd': np.zeros(len(k_range))})
        for j in range(len(k_range)) :
        
            # perform experiments
            out = [alg.Top_k(k_range[j], model) for i in range(N_samples)]
            out = np.array(out)
            
            # save data to file
            results.loc[j,'k']         = k_range[j]
            results.loc[j,'time_mn']   = np.mean([out[i,0] for i in range(N_samples)])
            results.loc[j,'rounds_mn'] = np.mean([out[i,1] for i in range(N_samples)])
            results.loc[j,'rounds_ind_mn'] = np.mean([out[i,2] for i in range(N_samples)])
            results.loc[j,'metric_mn'] = np.mean([out[i,3] for i in range(N_samples)])
            results.loc[j,'time_sd']   = np.std([out[i,0]  for i in range(N_samples)])
            results.loc[j,'rounds_sd'] = np.std([out[i,1]  for i in range(N_samples)])
            results.loc[j,'rounds_ind_sd'] = np.std([out[i,2] for i in range(N_samples)])
            results.loc[j,'metric_sd'] = np.std([out[i,3]  for i in range(N_samples)])
            results.to_csv('Top_k.csv', index = False)
            
            
    # ----- run TopK
    if Oblivious :
        print('----- testing Oblivious')
        results = pd.DataFrame(data = {'k': np.zeros(len(k_range)).astype('int'), 'time_mn': np.zeros(len(k_range)), 'rounds_mn': np.zeros(len(k_range)),'metric_mn': np.zeros(len(k_range)), 'time_sd': np.zeros(len(k_range)), 'rounds_sd': np.zeros(len(k_range)),'metric_sd': np.zeros(len(k_range))})
        for j in range(len(k_range)) :
        
            # perform experiments
            out = [alg.Oblivious(k_range[j], model) for i in range(N_samples)]
            out = np.array(out)
            
            # save data to file
            results.loc[j,'k']         = k_range[j]
            results.loc[j,'time_mn']   = np.mean([out[i,0] for i in range(N_samples)])
            results.loc[j,'rounds_mn'] = np.mean([out[i,1] for i in range(N_samples)])
            results.loc[j,'rounds_ind_mn'] = np.mean([out[i,2] for i in range(N_samples)])
            results.loc[j,'metric_mn'] = np.mean([out[i,3] for i in range(N_samples)])
            results.loc[j,'time_sd']   = np.std([out[i,0]  for i in range(N_samples)])
            results.loc[j,'rounds_sd'] = np.std([out[i,1]  for i in range(N_samples)])
            results.loc[j,'rounds_ind_sd'] = np.std([out[i,2] for i in range(N_samples)])
            results.loc[j,'metric_sd'] = np.std([out[i,3]  for i in range(N_samples)])
            results.to_csv('Oblivious.csv', index = False)

    # ----- run TopK
    if Random :
        print('----- testing Random')
        results = pd.DataFrame(data = {'k': np.zeros(len(k_range)).astype('int'), 'time_mn': np.zeros(len(k_range)), 'rounds_mn': np.zeros(len(k_range)),'metric_mn': np.zeros(len(k_range)), 'time_sd': np.zeros(len(k_range)), 'rounds_sd': np.zeros(len(k_range)),'metric_sd': np.zeros(len(k_range))})
        for j in range(len(k_range)) :
        
            # perform experiments
            out = [alg.Oblivious(k_range[j], model) for i in range(N_samples)]
            out = np.array(out)
            
            # save data to file
            results.loc[j,'k']         = k_range[j]
            results.loc[j,'time_mn']   = np.mean([out[i,0] for i in range(N_samples)])
            results.loc[j,'rounds_mn'] = np.mean([out[i,1] for i in range(N_samples)])
            results.loc[j,'rounds_ind_mn'] = np.mean([out[i,2] for i in range(N_samples)])
            results.loc[j,'metric_mn'] = np.mean([out[i,3] for i in range(N_samples)])
            results.loc[j,'time_sd']   = np.std([out[i,0]  for i in range(N_samples)])
            results.loc[j,'rounds_sd'] = np.std([out[i,1]  for i in range(N_samples)])
            results.loc[j,'rounds_ind_sd'] = np.std([out[i,2] for i in range(N_samples)])
            results.loc[j,'metric_sd'] = np.std([out[i,3]  for i in range(N_samples)])
            results.to_csv('Random.csv', index = False)
        
    # ----- run FAST_OMP
    if FAST_OMP :
        print('----- testing FAST_OMP')
        results = pd.DataFrame(data = {'k': np.zeros(len(k_range)).astype('int'), 'time_mn': np.zeros(len(k_range)), 'rounds_mn': np.zeros(len(k_range)),'metric_mn': np.zeros(len(k_range)), 'time_sd': np.zeros(len(k_range)), 'rounds_sd': np.zeros(len(k_range)),'metric_sd': np.zeros(len(k_range))})
        for j in range(len(k_range)) :
        
            # perform experiments
            out = [alg.FAST_OMP(model, k_range[j], eps_FAST_OMP, tau) for i in range(N_samples)]
            out = np.array(out)
            
            # save data to file
            results.loc[j,'k']         = k_range[j]
            results.loc[j,'time_mn']   = np.mean([out[i,0] for i in range(N_samples)])
            results.loc[j,'rounds_mn'] = np.mean([out[i,1] for i in range(N_samples)])
            results.loc[j,'rounds_ind_mn'] = np.mean([out[i,2] for i in range(N_samples)])
            results.loc[j,'metric_mn'] = np.mean([out[i,3] for i in range(N_samples)])
            results.loc[j,'time_sd']   = np.std([out[i,0]  for i in range(N_samples)])
            results.loc[j,'rounds_sd'] = np.std([out[i,1]  for i in range(N_samples)])
            results.loc[j,'rounds_ind_sd'] = np.std([out[i,2] for i in range(N_samples)])
            results.loc[j,'metric_sd'] = np.std([out[i,3]  for i in range(N_samples)])
            results.to_csv('FAST_OMP.csv', index = False)

    # ----- run SDS_MA
    if SDS_MA :
        print('----- testing SDS_MA')
        results = pd.DataFrame(data = {'k': np.zeros(len(k_range)).astype('int'), 'time': np.zeros(len(k_range)), 'rounds': np.zeros(len(k_range)),'metric': np.zeros(len(k_range))})
        alg.SDS_MA(model, k_range[-1])
        #for j in range(len(k_range)) :
        
            # perform experiments
        #    out = alg.SDS_MA(features, target, model, k_range[j])
        #    out = np.array(out)
            
            # save data to file
        #    results.loc[j,'k'] = k_range[j]
        #    results.loc[j,'time']   = out[0]
        #    results.loc[j,'rounds'] = out[1]
        #    results.loc[j,'metric'] = out[2]
        #    results.to_csv('SDS_MA.csv', index = False)

    
    return
    
    
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



# sample according to Gaussian distribution

def gaussianDistr(mean, Sigma, samples) :
    
    
    L = np.linalg.cholesky(Sigma)
    
    for i in range(samples) :
        
        mean = np.random.normal(loc=0.0, scale=1.0, size=L.shape[0])
        mean = np.dot(L, mean.transpose()) + mean
        if i == 0 : res = pd.DataFrame(mean).T
        else : res = res.append(pd.DataFrame(mean).T, ignore_index=True)
        #if i == 0 : write.to_csv('dataset.csv', index = False, header = False, mode = 'w')
        #else : write.to_csv('dataset.csv', index = False, header = False, mode = 'a')
    
    return res



# define features and target for the experiments

n_features = 100
n_samples  = 100

random.seed(1)



# create covariance matrix

mean = np.zeros(n_features+1)
Sigma = np.random.rand(n_features+1, n_features+1)
for i in range(Sigma.shape[0]) :
    for j in range(Sigma.shape[0]) :
        if i == j :
            Sigma[i, j] = 1.0
        else :
            Sigma[i, j] = 0.5
            Sigma[j, i] = Sigma[i,j]

# sample the features

df = pd.DataFrame(gaussianDistr(mean, Sigma, n_samples))
df.to_csv('healthdata.csv', index = False, header = False, mode = 'w')

target = df.iloc[:, 0]
features = df.iloc[:, range(1, Sigma.shape[0])]

# normalize target and features

target_norm = np.sum(target * target)
target = target / target_norm
features = features / target_norm


# initalize features and target

models.init_worker(features, target)

del(features)
del(target)

# choose if logistic or linear regression
model = 'linear'

# set range for the experiments
k_range = np.array([1,2,5,10])
k_range = np.array([1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
k_range = np.array([1, 1000, 2000, 3000])

# choose algorithms to be tested
SDS_OMP   = False 
FAST_OMP  = False
Top_k     = False 
SDS_MA    = False
Oblivious = True
Random    = True

# define parameters for algorithms

eps_FAST_OMP = 0.95
eps_DASH = 0.000000001

tau = 1.0
alpha = tau * tau
r = 1#mt.ceil(20/eps_DASH * np.log(features.shape[1])/ np.log(1 + eps_DASH/2))

# number of samples per evaluation
N_samples = 1 

# run experiment
run_experiment(model = model, k_range = k_range, eps_FAST_OMP = eps_FAST_OMP, tau = tau,eps_DASH = eps_DASH, alpha = alpha, r = r, N_samples = N_samples, SDS_OMP = SDS_OMP, SDS_MA = SDS_MA, Oblivious = Oblivious, FAST_OMP = FAST_OMP, Top_k = Top_k, Random = Random)
