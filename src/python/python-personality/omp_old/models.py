import sklearn.metrics as mt
import numpy as np
import pandas as pd
import time

from sklearn.linear_model import *



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
        log_loss = mt.log_loss(target, predict_prob)
        grad = np.dot(features.T, target - predictions)
        return grad, log_loss
      
    # do not conpute gradient of log likelihood
    else :
        log_loss = mt.log_loss(target, predict_prob)
        return log_loss



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
            #predict = np.ones((features.shape[0], 2)) * 0.5
            predict = (np.ones((features.shape[0])) * 0.5).reshape(features.shape[0], -1)

    # compute gradient of log likelihood
    if OMP :
        grad = np.dot(features.T, target - predict)
        return grad, score
     
    # do not compute gradient of log likelihood
    else : return score
