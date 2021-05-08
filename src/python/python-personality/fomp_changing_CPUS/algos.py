import functools
import math
import models
import random
import sys
import time

import multiprocessing as mp
import numpy as np
import pandas as pd

from joblib import Parallel, delayed, parallel_backend



# ------------------------------------------------------------------------------------------
#  additional functions
# ------------------------------------------------------------------------------------------

var_dict = {}

def init_worker(features, target):

    features = np.array(features)
    features_arr = mp.RawArray('d', features.shape[0]*features.shape[1])
    features_np = np.frombuffer(features_arr, dtype = np.dtype(float)).reshape(features.shape)
    np.copyto(features_np, features)
    
    target = np.array(target)
    target_arr = mp.RawArray('d', target.shape[0])
    target_np = np.frombuffer(target_arr, dtype = np.dtype(float)).reshape(target.shape)
    np.copyto(target_np, target)

    var_dict['features']       = features_np
    var_dict['features_shape'] = features.shape
    var_dict['target']         = target_np
    var_dict['target_shape']   = target.shape
 
 
 
def argmax_with_condition(arr, idx) :

    arr = np.array(arr)
    idx = np.array(idx)

    res = 0
    val = arr[0]
    
    for i in np.setdiff1d(range(len(arr)), idx) :
        if arr[i] >= val :
            res = i
            val = arr[i]
            
    return res
            


# ------------------------------------------------------------------------------------------
#  The FAST_OMP algorithm
# ------------------------------------------------------------------------------------------



def SAMPLE_SEQUENCE_parallel_oracle(i, A, S, model, N_CPU, k) :

    # get variables from shared array
    features_np = np.frombuffer(var_dict['features'], dtype = np.dtype(float)).reshape(var_dict['features_shape'])
    target_np   = np.frombuffer(var_dict['target'], dtype = np.dtype(float)).reshape(var_dict['target_shape'])
    
    # define vals
    point = []
    time = 0
    while i < len(A) :
        time += 1
        if models.constraint(features_np, target_np, np.append(S, A[0:i]), A[i], model, 'FAST_OMP', k) :
            point = np.append(point, i)
        else :
            break
        i += N_CPU
        
    return point, time
    
    
    
def SAMPLE_SEQUENCE(X, S, k, model, N_CPU) :
            
    # remove points from X that are in S
    S0 = S
    X = np.setdiff1d(X, S)
    rounds_ind = 0
        
    #print('-----')
    while True :
                
        # update random sequence
        #np.random.shuffle(X)
        #print(min(k - S.size, np.setdiff1d(X, S).size))
        A = np.random.choice(np.setdiff1d(X, S), min(k - S.size, np.setdiff1d(X, S).size), replace=False)
        if A.size == 0 : break
        
        # evaluate feasibility in parallel
        N_PROCESSES = min(N_CPU, len(A))
        
        pool = mp.Pool(N_PROCESSES)
        out = pool.map(functools.partial(SAMPLE_SEQUENCE_parallel_oracle, A = A, S = S, model = model, N_CPU = N_CPU, k = k), range(N_PROCESSES))
        out = np.array(out, dtype='object')
        rounds_ind += np.max(out[:, -1])
        pool.close()
        pool.join()
        
        # find which points to add
        feasible_points = np.array([])
        for i in range(N_PROCESSES) :
                feasible_points = np.append(feasible_points, out[i, 0])
        feasible_points = feasible_points.astype(int)
        feasible_points = np.sort(feasible_points)
        #print(feasible_points)
        
        if len(feasible_points) == 0 : break

        val = feasible_points[-1]
        for i in range(len(feasible_points) - 1) :
            if feasible_points[i] != feasible_points[i + 1] - 1 :
                val = feasible_points[i]
                break
        #print(val)
                
        # update current solution
        S = np.append(S, A[0:(val + 1)])

    return np.setdiff1d(S, S0), rounds_ind
            
            
        
        
def FAST_OMP_parallel_oracle(i, n_cpus, S, A, X_size, t, model, k) :

    # get variables from shared array
    features_np = np.frombuffer(var_dict['features'], dtype = np.dtype(float)).reshape(var_dict['features_shape'])
    target_np   = np.frombuffer(var_dict['target'], dtype = np.dtype(float)).reshape(var_dict['target_shape'])
    
    # define time for oracle calls
    rounds_ind = 0
        
    while i < len(A) :

        # compute gradiet
        out = np.array(models.oracle(features_np, target_np, np.append(S, A[0:i]), model, 'FAST_OMP'), dtype='object')
        vals = np.array([])
        
        # evaluate feasibility and compute Xj
        for j in np.setdiff1d(range(len(out[0])), np.append(S, A[0:i])) :
            rounds_ind += 1
            if models.constraint(features_np, target_np, np.append(S, A[0:i]), j, model, 'FAST_OMP', k) and out[0][j] >= pow(t, 0.5):
                vals = np.append(vals, j)
        Xj = [vals, out[1], i, False, rounds_ind]
        
        # return points if they fulfill cardinality condition
        if  Xj[0].size < X_size :
            Xj[-2] = True
            return np.array(Xj, dtype='object')
            
        # otherwise return the entire sequence
        elif i + n_cpus >= len(A) :
            return np.array(Xj, dtype='object')
        
        # update t
        i = i + n_cpus



def FAST_OMP(features, target, model, k, eps, tau, N_CPU) :

    '''
    The FAST_OMP algorithm, as in Algorithm 1 in the submission
    
    INPUTS:
    features -- the feature matrix
    target -- the observations
    model -- choose if the regression is linear or logistic
    k -- upper-bound on the solution size
    eps -- parameter epsilon for the approximation
    tau -- parameter m/M for the (M,m)-(restricted smootheness, restricted strong concavity)
    OUTPUTS:
    float run_time -- the processing time to optimize the function
    int rounds -- the number of parallel calls to the oracle function
    float metric -- a goodness of fit metric for the solution quality
    '''

    # define time and rounds
    run_time = time.time()
    rounds = 0
    rounds_ind = 0

    # define initial solution and number of outer iterations
    S = np.array([], int)
    iter = 0
    
    # copy features and target to shared array
    grad, metric = models.oracle(features, target, S, model, algo = 'FAST_OMP')
    init_worker(features, target)
    rounds += 1
    
    # redefine k
    feasible_sol = True
    
    while iter <= 1/eps and feasible_sol :

        # define new set X and copy to array
        X = np.setdiff1d(np.array(range(features.shape[1]), int), S)
        
        # find largest k-elements and update adaptivity
        if S.size != 0 :
            grad, metric = models.oracle(features, target, S, model, algo = 'FAST_OMP')
            rounds += 1
        
        # define parameter t
        t = np.power(grad, 2)
        t = np.sort(t)[::-1]
        t = (1 - eps) * tau *  np.sum(t[range(min(k, len(t)))]) / min(k, len(t))

        while X.size > 0 and feasible_sol :

            # define new random set of features and random index
            #A = np.random.choice(np.setdiff1d(X, S), min(k - S.size, np.setdiff1d(X, S).size), replace=False)
            A, new_rounds = SAMPLE_SEQUENCE(X, S, k, model, N_CPU)
            rounds_ind += new_rounds
            if len(A) == 0 :
                feasible_sol = False
                break

            # compute the increments in parallel
            X_size = (1 - eps) * X.size
            N_PROCESSES = min(N_CPU, len(A))
            
            pool = mp.Pool(N_PROCESSES)
            out = pool.map(functools.partial(FAST_OMP_parallel_oracle, n_cpus = N_PROCESSES, S = S, A = A, X_size = X_size, t = t, model = model, k = k), range(N_PROCESSES))
            pool.close()
            pool.join()
            
            out = np.array(out)
            
            # manually update rounds
            rounds += math.ceil(len(A)/N_PROCESSES)
            rounds_ind += max(out[:, -1])
                
            # compute sets Xj for the good incement
            idx = np.argsort(out[:, -3])
            for j in idx :
                if  out[j, -2] == True or j == idx[-1]:
                    S = np.append(S, A[0:(out[j, -3] + 1)])
                    S = (np.unique(S[S >= 0])).astype(int)
                    X = (out[j, 0]).astype(int)
                    metric = out[j, 1]
                    break

        iter = iter + 1;
        
    # update current time
    run_time = time.time() - run_time

    return run_time, rounds, rounds_ind, metric



# ------------------------------------------------------------------------------------------
#  The SDS_OMP algorithm
# ------------------------------------------------------------------------------------------



def SDS_OMP_parallel_oracle(A, S, model, k) :

    # get variables from shared array
    features_np = np.frombuffer(var_dict['features'], dtype = np.dtype(float)).reshape(var_dict['features_shape'])
    target_np   = np.frombuffer(var_dict['target'], dtype = np.dtype(float)).reshape(var_dict['target_shape'])
    
    # define vals
    point = []
    constraint = []
    for a in np.setdiff1d(A, S) :
        if models.constraint(features_np, target_np, S, a, model, 'SDS_OMP', k) :
            point = np.append(point, a)

    return point, len(np.setdiff1d(A, S))



def SDS_OMP(features, target, model, k, N_CPU) :

    '''
    The SDS_OMP algorithm, as in "Submodular Dictionary Selection for Sparse Representation", Krause and Cevher, ICML '10
    
    INPUTS:
    features -- the feature matrix
    target -- the observations
    model -- choose if the regression is linear or logistic
    k -- upper-bound on the solution size
    OUTPUTS:
    float run_time -- the processing time to optimize the function
    int rounds -- the number of parallel calls to the oracle function
    float metric -- a goodness of fit metric for the solution quality
    '''

    # multiprocessing
    init_worker(features, target)

    # define time and rounds
    run_time = time.time()
    rounds = 0
    rounds_ind = 0

    # define new solution
    S = np.array([], int)
    
    for idx in range(k) :
    
        # define and train model
        grad, metric = models.oracle(features, target, S, model, algo = 'SDS_OMP')
        rounds += 1
        
        # evaluate feasibility in parallel
        N_PROCESSES = min(N_CPU, len(grad))
        
        pool = mp.Pool(N_PROCESSES)
        out = pool.map(functools.partial(SDS_OMP_parallel_oracle, S = S, model = model, k = k), np.array_split(range(len(grad)), N_PROCESSES))
        pool.close()
        pool.join()
        
        out = np.array(out, dtype='object')
        rounds_ind += np.max(out[:, -1])

        # get feasible points
        points = np.array([])
        for i in range(N_PROCESSES) : points = np.append(points, np.array(out[i, 0]))
        points = points.astype('int')
        
        # break if points are no longer feasible
        if len(points) == 0 : break
        
        # otherwise add maximum point to current solution
        a = points[0]
        for i in points :
            if grad[i] > grad[a] :
                a = i
                
        if grad[a] >= 0 :
            S  = np.unique(np.append(S,i))
        else : break
        
    # update current time
    run_time = time.time() - run_time

    return run_time, rounds, metric
    
    
    
# ------------------------------------------------------------------------------------------
#  The Top_k algorithm
# ------------------------------------------------------------------------------------------

def TOP_K_parallel_oracle(X, model) :

    # get variables from shared array
    features_np = np.frombuffer(var_dict['features'], dtype = np.dtype(float)).reshape(var_dict['features_shape'])
    target_np   = np.frombuffer(var_dict['target'], dtype = np.dtype(float)).reshape(var_dict['target_shape'])
    
    # define vals
    res = pd.DataFrame(data = {'point': np.zeros(len(X)).astype('int'), 'metric': np.zeros(len(X))})

    idx = 0
    for a in X :
        out = models.oracle(features_np, target_np, a, model, 'SDS_MA')
        res.loc[idx,'point']  = int(a)
        res.loc[idx,'metric'] = out
        idx += 1
        
    return res

def Top_k(features, target, k, model,N_CPU) :

    '''
    This algorithm selects the k features with the highest score
    
    INPUTS:
    features -- the feature matrix
    target -- the observations
    model -- choose if the regression is linear or logistic
    k -- upper-bound on the solution size
    OUTPUTS:
    float run_time -- the processing time to optimize the function
    int rounds -- the number of parallel calls to the oracle function
    float metric -- a goodness of fit metric for the solution quality
    '''

    # measure time and adaptivity
    run_time = time.time()
    rounds = 0
    rounds_ind = 0
    
    # define ground set and initial solution
    S = np.array([], int)
    T = np.array([], int)
    X = np.array(range(features.shape[1]))
    
    # multiprocessing
    init_worker(features, target)
    
    # find marginal contributions
    marginal = np.array([])
    
    # evaluate points in parallel
    N_PROCESSES = min(N_CPU, len(X))
    
    pool = mp.Pool(N_PROCESSES)
    out = pool.map(functools.partial(TOP_K_parallel_oracle, model = model), np.array_split(X, N_PROCESSES))
    pool.close()
    pool.join()
    
    # manually update rounds
    #out = np.array(out) # this was added
    rounds += math.ceil(len(X)/N_PROCESSES)

    # postrpcess results
    point = np.array([])
    metric = np.array([])
    for i in range(N_PROCESSES) :
        point = np.append(point, np.array(out[i]['point']))
        metric = np.append(metric, np.array(out[i]['metric']))
    
    # add top k points to current solution
    while len(S) < k :
    
        # find maximum element that was not already added
        max = argmax_with_condition(metric, T)
        if models.constraint(features, target, S, int(point[max]), model, 'Top_k', k) :
            S = np.append(S, int(point[max]))
        T = np.append(T, int(point[max]))
        rounds_ind += 1
        
        if metric.size == len(T) : break
        
    # update current time
    run_time = time.time() - run_time
    
    # evaluate solution quality
    metric = models.oracle(features, target, S, model, 'Top_k')

    return run_time, rounds, rounds_ind, metric
    
    
# ------------------------------------------------------------------------------------------
#  The Oblivious2 Alg
# ------------------------------------------------------------------------------------------

def Oblivious2(features, target, k, model,N_CPU) : return
