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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
import optuna

# Load the dataset
data = pickle.load(open('wine_red_dataset.pkl', "rb"))
X = data['features']  # Features
Y = data['quality']   # Target variable
K = data['feature_names']  # Feature names

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Define the loss function
def Loss(y, y_pred):
    M = len(y)
    S = 0
    for i in range(M):
        S += (y[i] - y_pred[i]) ** 2
    return (1 / M) * S


# Define the gradient of the loss function
def dLoss_dW(x, y, y_pred):
    M = len(y)
    S = x.T @ (y - y_pred)
    return -2 * S / M

# Define the predict function
def predict(W, X):
    return X @ W

# Define the normal equation function
def normal_equation(X_train, y_train):
    inverse_x_xT = np.linalg.inv(X_train.T @ X_train)
    pseudo_inverse = inverse_x_xT @ X_train.T
    W = pseudo_inverse @ y_train
    return W

# Define the gradient descent function
def gradient_descent(X_train, y_train, X_test, y_test, learning_rate=0.0001, epochs=1000, loss_threshold=0.001):
    Wu = np.random.random(X_train.shape[1])  # Initial weight vector
    L_train = []
    L_test = []
    bestweights = Wu
    best_epoch = 0
    best_loss = float('inf')
    for i in range(epochs):
        Y_p = predict(Wu, X_train)
        Wu = Wu - learning_rate * dLoss_dW(X_train, y_train, Y_p)  # Update weights
        L_train.append(Loss(y_train, Y_p))
        L_test.append(Loss(y_test, predict(Wu, X_test)))
        if L_train[-1] + L_test[-1] < best_loss:
            best_loss = L_train[-1] + L_test[-1]
            bestweights = Wu
            best_epoch = i
        if L_train[-1] < loss_threshold:
            best_epoch = i
            break
    print('The best epoch is:', best_epoch)
    # Plot the loss
    epochs_axis = range(len(L_train))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_axis[100:], L_train[100:], label='Train loss')
    plt.plot(epochs_axis[100:], L_test[100:], label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Gradient Descent')
    plt.legend(fontsize=12, loc='upper right')
    plt.show()

    return bestweights, L_test[-1], L_train[-1]

# Define the get_batch function
def get_batch(X, y, batch_size=500):
    ix = np.random.choice(X.shape[0], batch_size)
    return X[ix, :], y[ix]

# Define the stochastic gradient descent function
def stochastic_gradient_descent(X_train, y_train, X_test, y_test, learning_rate=0.0001, epochs=1000, batch_size=500):
    Wu = np.random.random(X_train.shape[1])  # Initial weight vector
    L_train = []
    L_test = []
    Wu_best = Wu
    epoch_best = 0
    for i in range(epochs):
        X_batch, y_batch = get_batch(X_train, y_train, batch_size)
        Y_p = predict(Wu, X_batch)
        Wu = Wu - learning_rate * dLoss_dW(X_batch, y_batch, Y_p)  # Update weights
        L_train.append(Loss(y_batch, Y_p))
        L_test.append(Loss(y_test, predict(Wu, X_test)))

    # Plot the loss
    epochs_axis = range(len(L_train))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_axis[100:], L_train[100:], label='Train loss')
    plt.plot(epochs_axis[100:], L_test[100:], label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Stochastic Gradient Descent')
    plt.legend(fontsize=12, loc='upper right')
    plt.show()
    return Wu, L_test[-1], L_train[-1]

# Define the objective function for Optuna
def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    epochs = trial.suggest_int('epochs', 300, 600)
    _, loss_test, _ = gradient_descent(X_train, y_train, X_test, y_test, learning_rate=learning_rate, epochs=epochs)
    return loss_test

# Main function
def main():
    # Normal equation
    W_normal = normal_equation(X_train, y_train)

    pred_quality = predict(W_normal, X_test)
    NEL = Loss(y_test, pred_quality)
    print('#' * 100)
    print('\nNormal equation:')
    print('The loss on the test data using the normal equation is:', NEL)
    print('The weights using the normal equation are:', W_normal)


    # Gradient descent
    print('\n')
    print('#' * 100)
    print('\ngradient descent:')
    Wu, loss_test, loss_train = gradient_descent(X_train, y_train, X_test, y_test, learning_rate=0.0001, epochs=1500, loss_threshold=NEL + 0.05)
    print('The loss on the test data using the gradient descent method is:', loss_test)
    print('The loss on the training data using the gradient descent method is:', loss_train)


    # Stochastic gradient descent
    print('\n')
    print('#' * 100)
    print('\nstochastic gradient descent:')


    SWu, loss_test, loss_train = stochastic_gradient_descent(X_train, y_train, X_test, y_test, learning_rate=0.0001, epochs=1500, batch_size=len(X_train)//5)
    print('The loss on the test data using the stochastic gradient descent method is:', loss_test)
    print('The loss on the training data using the stochastic gradient descent method is:', loss_train)


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






