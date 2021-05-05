import functools
import math
import omp.models as models
import random
import sys
import time

import multiprocessing as mp
import numpy as np
import pandas as pd

from joblib import Parallel, delayed, parallel_backend

# new library
import threading
from threading import Thread



# ------------------------------------------------------------------------------------------
#  multi-threading functions
# ------------------------------------------------------------------------------------------



class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        #print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
        
        
        
def npThreads(target, args, variable) :

    # execute threads
    threads = []
    for i in variable:
    
        # run new thread
        input = (i,) + tuple(args)

        th = ThreadWithReturnValue(target=target, args=input)
        threads.append(th)
        th.start()
                
    # join threads
    out = []
    for th in threads : out.append(th.join())

    return np.array(out, dtype='object')



# ------------------------------------------------------------------------------------------
#  additional functions
# ------------------------------------------------------------------------------------------
 
 
 
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
    
    # define vals
    point = []
    time = 0
    while i < len(A) :
        time += 1
        if models.constraint(np.append(S, A[0:i]), A[i], model, 'FAST_OMP', k) :
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

    while True :
                
        # update random sequence
        A = np.random.choice(np.setdiff1d(X, S), min(k - S.size, np.setdiff1d(X, S).size), replace=False)
        if A.size == 0 : break
        
        # evaluate feasibility in parallel
        N_PROCESSES = min(N_CPU, len(A))
        x = range(N_PROCESSES)
        out = npThreads(target=SAMPLE_SEQUENCE_parallel_oracle, args=(A, S, model, N_CPU, k), variable = x)
        rounds_ind += np.max(out[:, -1])
        
        # find which points to add
        feasible_points = np.array([])
        for i in range(N_PROCESSES) :
                feasible_points = np.append(feasible_points, out[i, 0])
        feasible_points = feasible_points.astype(int)
        feasible_points = np.sort(feasible_points)

        
        if len(feasible_points) == 0 : break

        val = feasible_points[-1]
        for i in range(len(feasible_points) - 1) :
            if feasible_points[i] != feasible_points[i + 1] - 1 :
                val = feasible_points[i]
                break
                
        # update current solution
        S = np.append(S, A[0:(val + 1)])

    return np.setdiff1d(S, S0), rounds_ind
            
            
        
        
def FAST_OMP_parallel_oracle(i, n_cpus, S, A, X_size, t, model, k) :
    
    # define time for oracle calls
    rounds_ind = 0
        
    while i < len(A) :

        # compute gradiet
        out = np.array(models.oracle(np.append(S, A[0:i]), model, 'FAST_OMP'), dtype='object')
        vals = np.array([])
        
        # evaluate feasibility and compute Xj
        for j in np.setdiff1d(range(len(out[0])), np.append(S, A[0:i])) :
            rounds_ind += 1
            if models.constraint(np.append(S, A[0:i]), j, model, 'FAST_OMP', k) and out[0][j] >= pow(t, 0.5):
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



def FAST_OMP(model, k, eps, tau) :

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
    
    # get features shape
    features_np = np.frombuffer(models.var_dict['features'], dtype = np.dtype(float)).reshape(models.var_dict['features_shape'])
    X_shape = features_np.shape[1]
    del(features_np)
    
    # copy features and target to shared array
    grad, metric = models.oracle(S, model, algo = 'FAST_OMP')
    rounds += 1
    
    # redefine k
    feasible_sol = True
    
    # multiprocessing
    N_CPU = mp.cpu_count()
    
    while iter <= 1/eps and feasible_sol :

        # define new set X and copy to array
        X = np.setdiff1d(np.array(range(X_shape), int), S)
        
        # find largest k-elements and update adaptivity
        if S.size != 0 :
            grad, metric = models.oracle(S, model, algo = 'FAST_OMP')
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
            x = range(min(N_CPU, len(A)))
            out = npThreads(target=FAST_OMP_parallel_oracle, args=(N_PROCESSES, S, A, X_size, t, model, k), variable = x)
            
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
    
    # define vals
    point = []
    constraint = []
    for a in np.setdiff1d(A, S) :
        if models.constraint(S, a, model, 'SDS_OMP', k) :
            point = np.append(point, a)

    return point, len(np.setdiff1d(A, S))



def SDS_OMP(model, k) :

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
    N_CPU = mp.cpu_count()
    
    # save data to file
    results = pd.DataFrame(data = {'k': np.zeros(k).astype('int'), 'time': np.zeros(k), 'rounds': np.zeros(k),'metric': np.zeros(k)})

    # define time and rounds
    run_time = time.time()
    rounds = 0
    rounds_ind = 0

    # define new solution
    S = np.array([], int)
    
    for idx in range(k) :
    
        # define and train model
        grad, metric = models.oracle(S, model, algo = 'SDS_OMP')
        rounds += 1
        
        # evaluate feasibility in parallel
        N_PROCESSES = min(N_CPU, len(grad))
        x = np.array_split(range(len(grad)), N_PROCESSES)
        out = npThreads(target=SDS_OMP_parallel_oracle, args=(S, model, k), variable = x)

        # update adaptivity
        rounds_ind += np.max(out[:, -1])
        
        # save results to file
        results.loc[idx,'k']      = idx + 1
        results.loc[idx,'time']   = time.time() - run_time
        results.loc[idx,'rounds'] = int(rounds)
        results.loc[idx,'rounds_ind'] = rounds_ind
        results.loc[idx,'metric'] = metric
#        results.to_csv('SDS_OMP.csv', index = False)
        
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

        S  = np.unique(np.append(S,i))
        
    # update current time
    run_time = time.time() - run_time

    return run_time, rounds, metric
    
    
    
# ------------------------------------------------------------------------------------------
#  The SDS_MA algorithm
# ------------------------------------------------------------------------------------------



def SDS_MA_parallel_oracle(A, S, fS, model, k) :
    
    # define vals
    marginal = 0
    res = [0, 0, 0]

    for a in A :
        out = models.oracle(np.append(S, a), model, 'SDS_MA')
        idx = models.constraint(S, a, model, 'SDS_MA', k)
        if out - fS >= marginal and idx :
            res = [a, out, idx]
            marginal = out
             
    return res



def SDS_MA(model, k) :

    '''
    The SDS_MA algorithm, as in "Submodular Dictionary Selection for Sparse Representation", Krause and Cevher, ICML '10
    
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
    
    # save data to file
    results = pd.DataFrame(data = {'k': np.zeros(k).astype('int'), 'time': np.zeros(k), 'rounds': np.zeros(k),'metric': np.zeros(k)})

    # measure time and adaptivity
    run_time = time.time()
    rounds = 1
    rounds_ind = 0
    
    # define initial solution, ground set, and metric
    S = np.array([], int)
    features_np = np.frombuffer(models.var_dict['features'], dtype = np.dtype(float)).reshape(models.var_dict['features_shape'])
    X = np.array(range(features_np.shape[1]))
    del(features_np)
    metric = models.oracle(S, model, 'SDS_MA')
    
    # multiprocessing
    N_CPU = mp.cpu_count()
    
    for idx in range(k) :
    
        # define points that need to be added
        if X.shape == 0 : break

        # evaluate points in parallel
        N_PROCESSES = min(N_CPU, len(X))
        x = np.array_split(X, N_PROCESSES)
        out = npThreads(target=SDS_MA_parallel_oracle, args=(S, metric, model, k), variable = x)
        
        # manually update rounds
        rounds += math.ceil(len(X)/N_PROCESSES)
        rounds_ind += math.ceil(len(X)/N_PROCESSES)
        
        # save results to file
        results.loc[idx,'k']      = idx + 1
        results.loc[idx,'time']   = time.time() - run_time
        results.loc[idx,'rounds'] = int(rounds)
        results.loc[idx,'rounds_ind'] = int(rounds_ind)
        results.loc[idx,'metric'] = metric
        results.to_csv('SDS_MA.csv', index = False)

        # break if the solutions are not feasible
        if len(np.where(out[:, -1] == True)[0]) == 0 : break
        
        # otherwise find best solution and add it to set
        i = argmax_with_condition(out[:, 1], np.where(out[:, -1] == False)[0])
        
        # if the increment holds
        if out[i, -2] - metric >= 0 :
            metric = out[i, -2]
            S = np.append(S, int(out[i, 0]))
            S = np.unique(S[S >= 0])
            X = np.setdiff1d(X, int(out[i, 0]))
            
        # otherwise break
        else : break
    
    # update current time
    run_time = time.time() - run_time
    
    return run_time, rounds, rounds_ind, metric
    
    
    
# ------------------------------------------------------------------------------------------
#  The Top_k algorithm
# ------------------------------------------------------------------------------------------

def TOP_K_parallel_oracle(X, model) :
    
    # define vals
    res = pd.DataFrame(data = {'point': np.zeros(len(X)).astype('int'), 'metric': np.zeros(len(X))})

    idx = 0
    for a in X :
        out = models.oracle(a, model, 'SDS_MA')
        res.loc[idx,'point']  = int(a)
        res.loc[idx,'metric'] = out
        idx += 1
        
    res = np.array(res)
    return res

def Top_k(k, model) :

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
    
    features_np = np.frombuffer(models.var_dict['features'], dtype = np.dtype(float)).reshape(models.var_dict['features_shape'])
    X = np.array(range(features_np.shape[1]))
    del(features_np)
    
    # multiprocessing
    N_CPU = mp.cpu_count()
    
    # find marginal contributions
    marginal = np.array([])
    
    # evaluate points in parallel
    N_PROCESSES = min(N_CPU, len(X))
    x = np.array_split(X, N_PROCESSES)
    out = npThreads(target=TOP_K_parallel_oracle, args=(model,), variable = x)

    # manually update rounds
    rounds += math.ceil(len(X)/N_PROCESSES)

    # postrpcess results
    point = np.array([])
    metric = np.array([])
    for i in range(N_PROCESSES) :
        point = np.append(point, np.array(out[i, 0]))
        metric = np.append(metric, np.array(out[i,1]))
    
    # add top k points to current solution
    while len(S) < k :
    
        # find maximum element that was not already added
        max = argmax_with_condition(metric, T)
        if models.constraint(S, int(point[max]), model, 'Top_k', k) :
            S = np.append(S, int(point[max]))
        T = np.append(T, int(point[max]))
        rounds_ind += 1
        
        if metric.size == len(T) : break
        
    # update current time
    run_time = time.time() - run_time
    
    # evaluate solution quality
    metric = models.oracle(S, model, 'Top_k')
    return run_time, rounds, rounds_ind, metric
