import math
import omp.models as models
import random
import sys
import time

import numpy as np
import pandas as pd



# ------------------------------------------------------------------------------------------
#  The oracle function
# ------------------------------------------------------------------------------------------

def oracle(features, target, S, model, algo) :

    '''
    Train the model and outputs metric for a set of features
    
    INPUTS:
    features -- the feature matrix
    target -- the observations
    S -- index for the features used for model construction
    model -- choose if the regression is linear or logistic
    algo -- specify the output, based on the algorithm used for optimization

    OUTPUTS:
    float grad -- the garadient of the log-likelihood function
    float log_loss -- the log-loss for the trained model
    float -log_loss -- the negative log-loss, which is proportional to the log-likelihood
    float score -- the R^2 score for the trained linear model
    '''

    # preprocess current solution
    S = np.unique(S[S >= 0])
    # logistic model
    if model == 'logistic' :
    
        if algo == 'SDS_OMP' :
            grad, log_loss = models.Logistic_Regression(features, target, S, OMP = True)
            return grad, log_loss
        
        if algo == 'SDS_MA' or algo == 'Top_k' :
            log_loss = models.Logistic_Regression(features, target, S, OMP = False)
            return -log_loss
            
    # linear model
    if model == 'linear' :
    
        if algo == 'SDS_OMP' :
            grad, score = models.Linear_Regression(features, target, S, OMP = True)
            return grad, score
        
        if algo == 'SDS_MA' or algo == 'Top_k' :
            score = models.Linear_Regression(features, target, S, OMP = False)
            return score
            


# ------------------------------------------------------------------------------------------
#  The SDS_OMP algorithm
# ------------------------------------------------------------------------------------------

def SDS_OMP(features, target, model, k) :

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
     
    # define time and rounds
    run_time = time.time()
    rounds = 0
    # define new solution
    S = np.array([], int)
    
    for idx in range(k) :
    
        # define and train model
        grad, metric = oracle(features, target, S, model, algo = 'SDS_OMP')
        
        # update number of rounds
        rounds += 1
            
        # otherwise add points to current solution
        grad = np.power(grad, 2)
        grad[S] = sys.float_info.min
        
        # otherwise add points to S
        S  = np.unique(np.append(S, grad.argmax()))
        
    # update current time
    run_time = time.time() - run_time

    return run_time, rounds, metric
    
    
    
# ------------------------------------------------------------------------------------------
#  The SDS_MA algorithm
# ------------------------------------------------------------------------------------------

def SDS_MA(features, target, model, k) :

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

    # measure time and adaptivity
    run_time = time.time()
    rounds = 1
    # define initial solution, ground set, and metric
    S = np.array([], int)
    X = range(features.shape[1])
    metric = oracle(features, target, S, model, 'SDS_MA')

    #with Parallel(max_nbytes=1e6) as parallel :
    
    for idx in range(k) :
    
        # define points that need to be added
        X = np.setdiff1d(X, S)
        if X.shape == 0 : break

        for i in range(X.size) :
                
            # evaluate points in parallel
            out = oracle(features, target, np.append(S, X[i]), model, 'SDS_MA')
            #out = np.array(out)
                        
            # update number of rounds
            rounds += 1
                    
            # update solution quality
            if out >= metric :
                T = np.append(S, X[i])
                metric = out

        # update current solution
        S = T

    # evaluate solution quality
    if model == 'logistic' : metric = -metric
    
    # update current time
    run_time = time.time() - run_time
    
    return run_time, rounds, metric
    
    
    
# ------------------------------------------------------------------------------------------
#  The Top_k algorithm
# ------------------------------------------------------------------------------------------

def Top_k(features, target, k, model) :

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
    
    # define initial solution
    S = np.array([], int)
    
    # define ground set
    X = np.array(range(features.shape[1]))
    
    # define top marginal
    marginal = np.array([])
    
    # define points that need to be added
    for i in range(X.size) :
                
        # call oracle for each singleton
        out = oracle(features, target, X[i], model, 'Top_k')
        out = np.array(out)
                        
        # update number of rounds
        rounds += 1
            
        # update new marginal
        marginal = np.append(marginal, out)

    # find top k elements in marginal
    S = marginal.argsort()[-k:]
    
    # evaluate solution quality
    metric = oracle(features, target, S, model, 'Top_k')
    if model == 'logistic' : metric = -metric
    
    # update current time
    run_time = time.time() - run_time

    return run_time, rounds, metric
