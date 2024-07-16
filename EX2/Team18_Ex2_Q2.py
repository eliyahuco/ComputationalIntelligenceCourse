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
from time import time


data = pickle.load(file=open('wine_red_dataset.pkl', "rb"))
X = data['features'] # ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
Y = data['quality'] # [Quality]
K = data['feature_names'] # Strings of feature names

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

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
    This function calculates the prediction
    :param W: the weights
    :param X: the data
    :return: the prediction
    '''


    return X @ W
def normal_equation(X_train,y_train):
    '''
    This function calculates the weights using the normal equation without using the pinv function
    :param X_train: the training data
    :param y_train: the training labels
    :return: the weights
    '''
    inverse_x_xT = np.linalg.inv(X_train.T @ X_train)
    pseudo_inverse = inverse_x_xT @ X_train.T
    W = pseudo_inverse @ y_train
    return W

def gradient_descent(X_train,y_train,X_test,y_test,learning_rate = 0.001,epochs= 1000,loss_threshold = 0.001):
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
    bestweights = Wu
    best_epoch = 0
    best_loss = float('inf')
    for i in range(epochs):
        Y_p = predict(Wu, X_train)
        Wu = Wu - learning_rate * dLoss_dW(X_train, y_train ,Y_p) # Update weights
        L_train.append(Loss(y_train, Y_p))
        L_test.append(Loss(y_test, predict(Wu, X_test)))
        if L_train[-1] + L_test[-1] < best_loss:
            best_loss = L_train[-1] + L_test[-1]
            bestweights = Wu
            best_epoch = i
    print('The best epoch is:',best_epoch)
    return bestweights,L_test[-1],L_train[-1]


def get_batch(X, y, batch_size = 500):
    ix = np.random.choice(X.shape[0], batch_size)
    return X[ix, :], y[ix]

def stochastic_gradient_descent(X_train,y_train,X_test,y_test,learning_rate = 0.001,epochs= 1000,batch_size = len(X_train)//2):
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
import optuna

def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    epochs = trial.suggest_int('epochs', 300, 600)
    # Call your gradient descent function with the suggested learning_rate
    _, loss_test, loss_train = gradient_descent(X_train, y_train, X_test, y_test,
                                                learning_rate=learning_rate, epochs=epochs)

    # Return the evaluation metric (e.g., validation loss) as the objective value to minimize
    return loss_test  # Adjust based on whether you want to minimize or maximize
# Best learning rate found: 0.00023709581530500582
# Best number of epochs found: 1093
# Best loss found: 0.46359335974958604

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

    print('#'*100)
    Wu,loss_test,loss_train = gradient_descent(X_train,y_train,X_test,y_test,learning_rate=0.00001, epochs=5000, loss_threshold=0)
    print('The loss on the test data using the gradient descent method is:',loss_test)
    print('The loss on the training data using the gradient descent method is:',loss_train)
    print('The weights using the gradient descent method are:',Wu)
    print('#'*100)
    SWu,loss_test,loss_train = stochastic_gradient_descent(X_train,y_train,X_test,y_test,learning_rate=0.00001,epochs=5000,batch_size=len(X[0]//5))
    print('The loss on the test data using the stochastic gradient descent method is:',loss_test)
    print('The loss on the training data using the stochastic gradient descent method is:',loss_train)
    print('The weights using the stochastic gradient descent method are:',SWu)
    #
    # study = optuna.create_study(direction='minimize')  # Adjust direction based on your objective
    # study.optimize(objective, n_trials=30)
    #
    # best_params = study.best_params
    # best_learning_rate = best_params['learning_rate']
    # best_epochs = best_params['epochs']
    # print(f"Best learning rate found: {best_learning_rate}")
    # print(f"Best number of epochs found: {best_epochs}")
    # print(f"Best loss found: {study.best_value}")
    #
    #
    # # Now, use the best learning_rate to train your model
    # _, best_loss_test, best_loss_train = gradient_descent(X_train, y_train, X_test, y_test,
    #                                                       learning_rate=best_learning_rate, epochs=best_epochs)






if __name__ == '__main__':
    main()






