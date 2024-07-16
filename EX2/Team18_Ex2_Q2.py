"""Authors: Eliyahu cohen, id 304911084
            Daniel liberman, id 208206946

***both authors have contributed equally to the assignment.***
we were working together on the assignment and we both participated in the writing of the code and the writing of the report
---------------------------------------------------------------------------------
Short Description:

The task is to compute the weights of a linear model that relates 11 features of red wine samples to their quality score. This involves three methods:

Normal Equation (without using the pinv function).
Gradient Descent.
Stochastic Gradient Descent.

You need to:

Calculate the loss on the test data for each method.
Determine the number of epochs required for Gradient Descent to converge to results close to the Normal Equation.
Experiment with parameters like learning rate, batch size, and the number of epochs to achieve a satisfactory result.
Provide a summary of the performance of each method and assess whether a linear model is suitable for predicting wine quality in this specific case.


---------------------------------------------------------------------------------
"""
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import template4wine as t4w
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


data = pickle.load(file=open('wine_red_dataset.pkl', "rb"))
X = data['features'] # ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
Y = data['quality'] # [Quality]
K = data['feature_names'] # Strings of feature names

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



def Loss(y_test, y_pred):
    '''
    This function calculates the loss function
    :param y: the true values
    :param y_pred: the predicted values
    :return: the loss value
    '''
    M = len(y_test)
    S = 0
    for i in range(M):
        S += (y_test[i] - y_pred[i])**2
    return (1/M) * S

def dLoss_dW(x, y, y_pred):
    '''
    This function calculates the derivative of the loss function
    :param x: the input data
    :param y: the true values
    :param y_pred: the predicted values
    :return: the derivative of the loss function
    '''
    M = len(y)
    S = 0
    for i in range(M):
        S += -x[i] * (y[i] - y_pred[i])
    return (2/M) * S

def predict(W, X):
    '''
    This function predicts the values of the output
    :param W: the weights
    :param X: the input data
    :return: the predicted values
    '''
    Y_p = []
    for x in X:
        Y_p.append(np.matmul(W,x))
    return np.array(Y_p)

def normal_equation(X_train,y_train):
    '''
    This function calculates the weights using the normal equation without using the pinv function
    :param X_train: the training data
    :param y_train: the training labels
    :return: the weights
    '''
    x = np.array(X_train)
    y = np.array(y_train)
    inverse_x_xT = np.linalg.inv(np.matmul(x.T,x))
    pseudo_inverse = np.matmul(inverse_x_xT,x.T)
    W = np.matmul(pseudo_inverse,y)
    return W

def gradient_descent(X_train,y_train,X_test,y_test,learning_rate = 0.1,epochs= 1000):
    '''
    This function calculates the weights using the gradient descent method
    :param X_train: the training data
    :param y_train: the training labels
    :param X_test: the test data
    :param y_test: the test labels
    :param learning_rate: the learning rate
    :param epochs: the number of epochs
    :return: the weights, the loss on the test data, the loss on the training data
    '''
    Wu = np.random.randn(X_train.shape[1]) # Initial weigth vector
    L_train = []
    L_test = []
    for i in range(epochs):
        Y_p = predict(Wu, X_train)
        Wu = Wu - learning_rate * dLoss_dW(X_train, y_train ,Y_p) # Update weights
        L_train.append(Loss(y_train, Y_p))
        L_test.append(Loss(y_test, predict(Wu, X_test)))
    return Wu,L_test[-1],L_train[-1]


def get_batch(X, y, batch_size = 500):
    ix = np.random.choice(X.shape[0], batch_size)
    return X[ix, :], y[ix]

def stochastic_gradient_descent(X_train,y_train,X_test,y_test,learning_rate = 0.1,epochs= 1000,batch_size = len(X_train)//2):
    '''
    This function calculates the weights using the stochastic gradient descent method
    :param X_train: the training data
    :param y_train: the training labels
    :param X_test: the test data
    :param y_test: the test labels
    :param learning_rate: the learning rate
    :param epochs: the number of epochs
    :param batch_size: the size of the batch
    :return: the weights, the loss on the test data, the loss on the training data
    '''
    Wu = np.random.randn(X_train.shape[1]) # Initial weigth vector
    L_train = []
    L_test = []
    for i in range(epochs):
        X_batch, y_batch = get_batch(X_train, y_train,batch_size)
        Y_p = predict(Wu, X_batch)
        Wu = Wu - learning_rate * dLoss_dW(X_batch, y_batch ,Y_p) # Update weights
        L_train.append(Loss(y_batch, Y_p))
        L_test.append(Loss(y_test, predict(Wu, X_test)))
    return Wu,L_test[-1],L_train[-1]

def main():
    '''
    This function runs the main code
    '''
    W_normal = normal_equation(X_train,y_train)
    pred_quality = predict(W_normal,X_test)
    NEL = Loss(y_test,pred_quality)
    print('#'*100)
    print('The loss on the test data using the normal equation is:',NEL)
    print('The weights using the normal equation are:',W_normal)
    w2 = sklearn.linear_model.LinearRegression().fit(X_train,y_train)
    print('\nThe weights using the sklearn linear regression are:',w2.coef_)
    nel2 = sklearn.metrics.mean_squared_error(y_test,w2.predict(X_test))
    print('The loss on the test data using the sklearn linear regression is:',nel2)
    # print('#'*100)
    # Wu,loss_test,loss_train = gradient_descent(X_train,y_train,X_test,y_test,learning_rate=0.001,epochs=200)
    # print('The loss on the test data using the gradient descent method is:',loss_test)
    # print('The loss on the training data using the gradient descent method is:',loss_train)
    # print('The weights using the gradient descent method are:',Wu)
    # print('#'*100)
    # SWu,loss_test,loss_train = stochastic_gradient_descent(X_train,y_train,X_test,y_test,learning_rate=0.001,epochs=300,batch_size=200)
    # print('The loss on the test data using the stochastic gradient descent method is:',loss_test)
    # print('The loss on the training data using the stochastic gradient descent method is:',loss_train)
    # print('The weights using the stochastic gradient descent method are:',SWu)

# [ 9.77489605e-03 -1.01175343e+00 -1.42358272e-01  3.68049243e-04
#  -1.82425404e+00  5.77724460e-03 -3.68605445e-03  4.29221111e+00
#  -4.64653031e-01  8.22089195e-01  2.95314313e-01]


if __name__ == '__main__':
    main()






