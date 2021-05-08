import sklearn.metrics as mt
import math
import numpy as np
import pandas as pd

from sklearn.linear_model import *

def constraint(features_np, target_np, S, a, model, algo, k) : #return True

    N_GROUPS = 5
    STEP = math.floor(features_np.shape[1]/N_GROUPS)
    #J = np.array_split(range(features_np.shape[1]), N_GROUPS)
    #J = range(0, features_np.shape[1] , )
    res = True
    i = 0
    while i < features_np.shape[1] :
        j = min(i + STEP, features_np.shape[1] - 1)
        if len((np.where((S >= i) & (S <= j) ))[0]) >= math.ceil((j - i) * k) :
            res = False
            break
        i += STEP

    return True


def oracle(features_np, target_np, S, model, algo) :

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
            grad, log_like = Logistic_Regression(features_np, target_np, S, OMP = True)
            return grad, log_like
        
        if algo == 'SDS_MA' or algo == 'DASH' or algo == 'Top_k' :
            log_like = Logistic_Regression(features_np, target_np, S, OMP = False)
            return log_like
            
    # linear model
    if model == 'linear' :
    
        if algo == 'FAST_OMP' or algo == 'SDS_OMP' :
            grad, score = Linear_Regression(features_np, target_np, S, OMP = True)
            return grad, score
        
        if algo == 'SDS_MA' or algo == 'DASH' or algo == 'Top_k' :
            score = Linear_Regression(features_np, target_np, S, OMP = False)
            return score



# ------------------------------------------------------------------------------------------
#  logistic regression
# ------------------------------------------------------------------------------------------

def Logistic_Regression(features, target, dims, OMP = True):

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

    # preprocess features
    features = pd.DataFrame(features)

    if not (features.iloc[:,dims]).empty :
    
        # define sparse features
        sparse_features = np.array(features.iloc[:,dims])
        if sparse_features.ndim == 1 : sparse_features = sparse_features.reshape(sparse_features.shape[0], 1)
        
        # get model, predict probabilities, and predictions
        model = LogisticRegression(max_iter = 10000).fit(sparse_features , target)
        predict_prob  = np.array(model.predict_proba(sparse_features))
        if OMP : predictions = model.predict(sparse_features)
        
    else :
    
        # predict probabilities, and predictions
        predict_prob  = np.ones((features.shape[0], 2)) * 0.5
        if OMP : predictions = np.ones((features.shape[0])) * 0.5

    # conpute gradient of log likelihood
    if OMP :
        log_like = (-mt.log_loss(target, predict_prob) + mt.log_loss(target, np.ones((features.shape[0], 2)) * 0.5)) * len(target)
        grad = np.dot(features.T, target - predictions)
        return grad, log_like
      
    # do not conpute gradient of log likelihood
    else :
        log_like = (-mt.log_loss(target, predict_prob) + mt.log_loss(target, np.ones((features.shape[0], 2)) * 0.5)) * len(target)
        return log_like



# ------------------------------------------------------------------------------------------
#  linear regression
# ------------------------------------------------------------------------------------------

def Linear_Regression(features, target, dims, OMP = True):

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

    # preprocess features and target
    features = pd.DataFrame(features)
    target = np.array(target).reshape(target.shape[0], -1)
    
    if not (features.iloc[:,dims]).empty :
    
        # define sparse features
        sparse_features = np.array(features.iloc[:,dims])
        if sparse_features.ndim == 1 : sparse_features = sparse_features.reshape(sparse_features.shape[0], 1)

        # get model, predict probabilities, and predictions
        model = LinearRegression().fit(sparse_features , target)
        score = model.score(sparse_features , target)
        if OMP : predict = model.predict(sparse_features)
        
    else :
    
        # predict probabilities, and predictions
        score = 0
        if OMP :
            predict = (np.ones((features.shape[0])) * 0.5).reshape(features.shape[0], -1)

    # compute gradient of log likelihood
    if OMP :
        grad = np.dot(features.T, target - predict)
        return grad, score
     
    # do not compute gradient of log likelihood
    else : return score
