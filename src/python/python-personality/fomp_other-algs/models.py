import multiprocessing as mp
import sklearn.metrics as mt
import math
import numpy as np
import pandas as pd

from sklearn.linear_model import *

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
    

def constraint(S, a, model, algo, k) :

    #get the features
    features_np = np.frombuffer(var_dict['features'], dtype = np.dtype(float)).reshape(var_dict['features_shape'])
    length = features_np.shape[1]
    
    # define number of groups
    N_GROUPS = 5
    STEP = math.floor(length/N_GROUPS)

    # run algo
    res = True
    i = 0
    while i < length :
        j = min(i + STEP, length - 1)
        if len((np.where((S >= i) & (S <= j) ))[0]) >= math.ceil((j - i) * k) :
            res = False
            break
        i += STEP

    res = True
    return True


def oracle(S, model, algo) :

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
    
        if algo == 'FAST_OMP' or algo == 'SDS_OMP' :
            grad, log_like = Logistic_Regression(S, OMP = True)
            return grad, log_like
        
        if algo == 'SDS_MA' or algo == 'DASH' or algo == 'Top_k' :
            log_like = Logistic_Regression(S, OMP = False)
            return log_like
            
    # linear model
    if model == 'linear' :
    
        if algo == 'FAST_OMP' or algo == 'SDS_OMP' :
            grad, score = Linear_Regression(S, OMP = True)
            return grad, score
        
        if algo == 'SDS_MA' or algo == 'DASH' or algo == 'Top_k' :
            score = Linear_Regression(S, OMP = False)
            return score



# ------------------------------------------------------------------------------------------
#  logistic regression
# ------------------------------------------------------------------------------------------

def Logistic_Regression(dims, OMP = True):

    '''
    Logistic regression for a given set of features
    
    INPUTS:
    features -- the feature matrix
    target -- the observations
    dims -- index for the features used for model construction
    OMP -- if set to TRUE the function returns grad
    OUTPUTS:
    float grad -- the garadient of the log-likelihood function
    float log_loss -- the log-loss for the trained model
    '''
    
    # get variables from shared array
    features_np = np.frombuffer(var_dict['features'], dtype = np.dtype(float)).reshape(var_dict['features_shape'])
    target_np   = np.frombuffer(var_dict['target'], dtype = np.dtype(float)).reshape(var_dict['target_shape'])

    # preprocess features
    features_np = pd.DataFrame(features_np)

    if not (features_np.iloc[:,dims]).empty :
    
        # define sparse features
        sparse_features = np.array(features_np.iloc[:,dims])
        if sparse_features.ndim == 1 : sparse_features = sparse_features.reshape(sparse_features.shape[0], 1)
        
        # get model, predict probabilities, and predictions
        model = LogisticRegression(max_iter = 10000).fit(sparse_features , target_np)
        predict_prob  = np.array(model.predict_proba(sparse_features))
        if OMP : predictions = model.predict(sparse_features)
        
    else :
    
        # predict probabilities, and predictions
        predict_prob  = np.ones((features_np.shape[0], 2)) * 0.5
        if OMP : predictions = np.ones((features_np.shape[0])) * 0.5

    # conpute gradient of log likelihood
    if OMP :
        log_like = (-mt.log_loss(target_np, predict_prob) + mt.log_loss(target_np, np.ones((features_np.shape[0], 2)) * 0.5)) * len(target_np)
        grad = np.dot(features_np.T, target_np - predictions)
        return grad, log_like
      
    # do not conpute gradient of log likelihood
    else :
        log_like = (-mt.log_loss(target_np, predict_prob) + mt.log_loss(target_np, np.ones((features_np.shape[0], 2)) * 0.5)) * len(target_np)
        return log_like



# ------------------------------------------------------------------------------------------
#  linear regression
# ------------------------------------------------------------------------------------------

def Linear_Regression(dims, OMP = True):

    '''
    Linear regression for a given set of features
    
    INPUTS:
    features -- the feature matrix
    target -- the observations
    dims -- index for the features used for model construction
    OMP -- if set to TRUE the function returns grad

    OUTPUTS:
    float grad -- the garadient of the log-likelihood function
    float score -- the R^2 score for the trained model
    '''
    
    # get variables from shared array
    features_np = np.frombuffer(var_dict['features'], dtype = np.dtype(float)).reshape(var_dict['features_shape'])
    target_np   = np.frombuffer(var_dict['target'], dtype = np.dtype(float)).reshape(var_dict['target_shape'])

    # preprocess features and target
    features_np = pd.DataFrame(features_np)
    target_np = np.array(target_np).reshape(target_np.shape[0], -1)
    
    if not (features_np.iloc[:,dims]).empty :
    
        # define sparse features
        sparse_features = np.array(features_np.iloc[:,dims])
        if sparse_features.ndim == 1 : sparse_features = sparse_features.reshape(sparse_features.shape[0], 1)

        # get model, predict probabilities, and predictions
        model = LinearRegression().fit(sparse_features , target_np)
        score = model.score(sparse_features , target_np)
        if OMP : predict = model.predict(sparse_features)
        
    else :
    
        # predict probabilities, and predictions
        score = 0
        if OMP :
            #predict = (np.ones((features.shape[0])) * 0.5).reshape(features.shape[0], -1)
            predict = (np.zeros((features_np.shape[0]))).reshape(features_np.shape[0], -1)

    # compute gradient of log likelihood
    if OMP :
        grad = np.dot(features_np.T, target_np - predict)
        return grad, score
     
    # do not compute gradient of log likelihood
    else : return score
