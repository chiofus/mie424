
# NAME: Durango González Avitia
# UTorID: gonza574
# UTorEmail: durango.gonzalez@mail.utoronto.ca
# Student # 1007630459

################
## MIE424 (2026 Winter)
##
## Code for Problem 4 Assignment 2
##
## INSTRUCTIONS:
## The following file fits different least square regression by
## projected gradient descent
##
## Fill in the code in parts labeled "FILL IN".
##
################

import numpy as np
from sklearn.model_selection import train_test_split
from typing import List
from numpy import ndarray
# Fixing seed for evaluation
seed=4242026
np.random.seed(seed)
###################################
# For part (a) to part (c)
###################################

# Generate Data
def generate_data(num_samples, num_features, num_nonneg, nonneg_value):

    # generate simulation data with first num_nonneg coordinates being positive
    beta_star = np.random.randn(num_features)
    beta_star[0:num_nonneg] = nonneg_value

    X = np.random.randn(num_samples, num_features)
    Y = np.matmul(X,beta_star)+np.random.randn(num_samples)

    return X, Y, beta_star


# Projection Function
def nonneg_project(u: ndarray, S: ndarray):
    ##
    ## Computes
    ##   argmin_{v}  || u - v ||_2^2
    ##   subject to: v[j] >= 0 for all j in S
    ##
    ## note that u is not changed after the projection
    ##
    ## FILL IN:
    ##
    ##
    ##

    v: ndarray = [val for val in u] #this would be optimal if all u were positive

    #now, check and fix for v[j] in S
    for indx in S:
        if v[indx] < 0: v[indx] = 0 #setting to zero minimizes the increase to obj fn.

    return v

def get_beta_ols(X: ndarray, Y: ndarray):
    #Want to find what an 'optimal' least squares beta approximation would be

    #Given by: (X^TX)^(-1)X^TY

    return np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)

def get_L2_norm_squared(X: ndarray):
    return np.dot(X,X)

# Train the model with projected gradient descent
def nonneg_OLS(X, Y, S, threshold=1e-22):
    n, d = X.shape
    
    ## XTX: matrix multiplication of X^T and X
    ## XTY: matrix multiplication of X^T and Y
    ## L:   maximum eigenvalue of the matrix X^TX

    XTX  = np.matmul(np.transpose(X),X)
    XTY  = np.matmul(np.transpose(X),Y)
    w, v = np.linalg.eig(XTX)
    L    = np.max(w)

    # Initialize beta with zeros
    beta = np.zeros(d)

    #Because of the projection onto C, I noticed that some beta weights simply stay too large, and the model never converges for
    #the given learning rate, so I decided to also include a max nbr. of iters, in case the learning threshold is never reached.
    curr_iter = 0

    while True:
        curr_iter += 1

        ## FILL IN: compute beta_new using projected gradient descent
        ## use 1/L as learning rate
        ## Loss function: 1/2 * || X beta - Y ||_2^2

        #First, need to find gradients vector for current beta's loss
        #Vector of loss' gradients is given by: (X^TX)Beta-(X^TY)

        beta_loss_grad: ndarray = np.matmul(XTX, beta)-XTY

        #We can now find the updated beta by adding the loss' grad times our learning rate 1/L
        beta_updated: ndarray = beta - (1/L)*beta_loss_grad

        #Now, simply project it onto C for the entries that must be greater than zero, according to S
        beta_new = nonneg_project(beta_updated, S)

        ## FILL IN: complete the stopping criteria
        ## threshold: if the update in beta has squared l2 norm less than threshold then stop the gradient descent
        update_size = get_L2_norm_squared((1/L)*beta_loss_grad)
        if (update_size < threshold or curr_iter > 10000):
            break
        else:
            beta = beta_new
    return beta_new


num_trials = 300
num_samples  = 100
num_features = 20
num_nonneg   = 7
nonneg_value = 0.05

ols_beta_err    = np.zeros(num_trials)
ols2_beta_err   = np.zeros(num_trials)
nonneg_beta_err = np.zeros(num_trials)

ols_train_err    = np.zeros(num_trials)
ols2_train_err   = np.zeros(num_trials)
nonneg_train_err = np.zeros(num_trials)

ols_test_err    = np.zeros(num_trials)
ols2_test_err   = np.zeros(num_trials)
nonneg_test_err = np.zeros(num_trials)

for i in range(num_trials):

    X, Y, beta_star = generate_data(num_samples, num_features, num_nonneg, nonneg_value)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)

    indices_nonneg = np.arange(num_nonneg)

    ## FILL IN: compute beta_ols, beta_nonneg
    ## beta_ols: use analytical formula to compute beta_ols
    ## beta_nonneg: use the function nonneg_OLS to compute beta_nonneg

    #Assume that these tree estimators are to only be generated using the X_train and Y_train subsets of X and Y, respectively.

    beta_ols    = get_beta_ols(X_train, Y_train)
    beta_ols2   = nonneg_project(beta_ols,indices_nonneg)
    beta_nonneg = nonneg_OLS(X_train, Y_train, indices_nonneg)

    ## FILL IN: compute estimation error, mean squared error of predictions
    ## _beta_err[i]: mean squared error of estimating beta_true in ith trial
    ## _train_err[i]: mean squared error of prediction on the training set using the model trained
    ## _test_err[i]: mean squared error of prediction on the test set using the model trained
    ##
    ## Hint: for prediciton error, normalize your result for them to be comparable

    ols_beta_err[i]     = get_L2_norm_squared(beta_ols - beta_star)
    ols2_beta_err[i]    = get_L2_norm_squared(beta_ols2 - beta_star)
    nonneg_beta_err[i]  = get_L2_norm_squared(beta_nonneg - beta_star)

    ols_train_err[i]    = (1/2)*get_L2_norm_squared(np.matmul(X_train, beta_ols) - Y_train)
    ols2_train_err[i]   = (1/2)*get_L2_norm_squared(np.matmul(X_train, beta_ols2) - Y_train)
    nonneg_train_err[i] = (1/2)*get_L2_norm_squared(np.matmul(X_train, beta_nonneg) - Y_train)

    ols_test_err[i]     = (1/2)*get_L2_norm_squared(np.matmul(X_test, beta_ols) - Y_test)
    ols2_test_err[i]    = (1/2)*get_L2_norm_squared(np.matmul(X_test, beta_ols2) - Y_test)
    nonneg_test_err[i]  = (1/2)*get_L2_norm_squared(np.matmul(X_test, beta_nonneg) - Y_test)

print(f'OLS Average             beta Estimation MSE: {np.average(ols_beta_err):.3f}   Average Train MSE: {np.average(ols_train_err):.3f}   Average Test MSE: {np.average(ols_test_err):.3f} ')
print(f'Projected OLS Average   beta Estimation MSE: {np.average(ols2_beta_err):.3f}   Average Train MSE: {np.average(ols2_train_err):.3f}   Average Test MSE: {np.average(ols2_test_err):.3f} ')
print(f'Non-negative LS Average beta Estimation MSE: {np.average(nonneg_beta_err):.3f}   Average Train MSE: {np.average(nonneg_train_err):.3f}   Average Test MSE: {np.average(nonneg_test_err):.3f} ')

###################################
# For part (d)
###################################
# Generate Data
def generate_data_2(num_samples, num_features):

    beta_star = np.random.randn(num_features)
    beta_star = beta_star / np.linalg.norm(beta_star)

    X = np.random.randn(num_samples, num_features)
    Y = np.matmul(X,beta_star)+np.random.randn(num_samples)
    return X, Y, beta_star

# Projection Function
def unit_ball_project(u):
    ## Computes
    ##   argmin_{v}  || u - v ||_2^2
    ##   subject to: ||v||_2<=1
    ##
    ## note that u is not changed after the projection
    ##
    ##
    ## FILL IN:
    ##
    ##
    ##
    return v

# Train the model with projected gradient descent
def unit_ball_CLS(X, Y, threshold=1e-22):
    ## Input
    ## X: X_train
    ## Y: Y_train
    ## threshold: if the update in beta has squared l2 norm less than threshold then stop the gradient descent
    ## Output
    ## beta_cls: the trained estimate for beta using projected gradeint descent
    ## L:   maximum eigenvalue of the matrix X^TX
    ## 1/L: learning rate for projected gradient descent
    ## Loss function: 1/2 * || X beta - Y ||_2^2
    
    XTX  = np.matmul(np.transpose(X),X)
    XTY  = np.matmul(np.transpose(X),Y)
    w, v = np.linalg.eig(XTX)
    L    = np.max(w)
    
    ## FILL IN: complete the function
    
    
    
    
    
    

    return beta_cls

num_trials = 300
num_samples  = 100
num_features = 20

ols_beta_err  = np.zeros(num_trials)
ols2_beta_err = np.zeros(num_trials)
cls_beta_err  = np.zeros(num_trials)

ols_train_err  = np.zeros(num_trials)
ols2_train_err = np.zeros(num_trials)
cls_train_err  = np.zeros(num_trials)

ols_test_err  = np.zeros(num_trials)
ols2_test_err = np.zeros(num_trials)
cls_test_err  = np.zeros(num_trials)

np.random.seed(seed)

for i in range(num_trials):

    X, Y, beta_star = generate_data_2(num_samples, num_features)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)

    ## FILL IN: compute beta_ols, beta_nonneg
    ## beta_ols: use analytical formula to compute beta_ols
    ## beta_ols2: projection beta_ols using function unit_ball_project(...)
    ## beta_cls: use the function unit_ball_cls to compute beta_nonneg

    beta_ols    = None
    beta_ols2   = None
    beta_nonneg = None

    ## FILL IN: compute estimation error, mean squared error of predictions
    ## _beta_err[i]: mean squared error of estimating beta_true in ith trial
    ## _train_err[i]: mean squared error of prediction on the training set using the model trained
    ## _test_err[i]: mean squared error of prediction on the test set using the model trained
    ##
    ## Hint: for prediciton error, normalize your result for them to be comparable

    ols_beta_err[i]  = None
    ols2_beta_err[i] = None
    cls_beta_err[i]  = None

    ols_train_err[i]  = None
    ols2_train_err[i] = None
    cls_train_err[i]  = None

    ols_test_err[i]  = None
    ols2_test_err[i] = None
    cls_test_err[i]  = None

print(f'OLS Average             beta Estimation MSE: {np.average(ols_beta_err):.3f}   Average Train MSE: {np.average(ols_train_err):.3f}   Average Test MSE: {np.average(ols_test_err):.3f} ')
print(f'Projected OLS Average   beta Estimation MSE: {np.average(ols2_beta_err):.3f}   Average Train MSE: {np.average(ols2_train_err):.3f}   Average Test MSE: {np.average(ols2_test_err):.3f} ')
print(f'Constrained LS Average  beta Estimation MSE: {np.average(cls_beta_err):.3f}   Average Train MSE: {np.average(cls_train_err):.3f}   Average Test MSE: {np.average(cls_test_err):.3f} ')
