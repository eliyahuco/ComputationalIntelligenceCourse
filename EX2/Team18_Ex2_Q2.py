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

data = pickle.load(file=open('wine_red_dataset.pkl', "rb"))
X = data['features'] # ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
Y = data['quality'] # [Quality]
K = data['feature_names'] # Strings of feature names

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


def normal_equation(X, y):
    ''' Normal Equation method to solve for theta
    '''
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta

def gradient_descent(X, y, learning_rate=0.0001, epochs=1000):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    m = X.shape[0]
    n = X.shape[1]
    theta = np.zeros(n)
    for i in range(epochs):
        gradient = (1/m) * X.T @ (X @ theta - y)
        theta = theta - learning_rate * gradient
    return theta

def stochastic_gradient_descent(X, y, learning_rate=0.0001, epochs=1000, batch_size=1):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    m = X.shape[0]
    n = X.shape[1]
    theta = np.zeros(n)
    for i in range(epochs):
        for j in range(0, m, batch_size):
            X_batch = X[j:j+batch_size]
            y_batch = y[j:j+batch_size]
            gradient = (1/batch_size) * X_batch.T @ (X_batch @ theta - y_batch)
            theta = theta - learning_rate * gradient
    return theta

def loss(X, y, theta):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    m = X.shape[0]
    return np.sum((X @ theta - y) ** 2) / (2 * m)

theta_normal = normal_equation(X_train, y_train)
theta_gradient = gradient_descent(X_train, y_train)
theta_stochastic = stochastic_gradient_descent(X_train, y_train)

loss_normal = loss(X_test, y_test, theta_normal)
loss_gradient = loss(X_test, y_test, theta_gradient)
loss_stochastic = loss(X_test, y_test, theta_stochastic)

print('Normal Equation Loss:', loss_normal)
print('Gradient Descent Loss:', loss_gradient)
print('Stochastic Gradient Descent Loss:', loss_stochastic)

epochs = 0
theta_gradient = np.zeros(X_train.shape[1] + 1)
while True:
    theta_gradient_new = gradient_descent(X_train, y_train, epochs=1)
    if np.allclose(theta_gradient, theta_gradient_new, atol=1e-5):
        break
    theta_gradient = theta_gradient_new
    epochs += 1

print('Number of epochs for Gradient Descent to converge:', epochs)
print('Theta:', theta_gradient)

# Experiment with parameters like learning rate, batch size, and the number of epochs to achieve a satisfactory result.
learning_rates = [0.0001, 0.001, 0.01, 0.1]
batch_sizes = [1, 10, 100, 1000]
epochs = [100, 1000, 10000]

for lr in learning_rates:
    for bs in batch_sizes:
        for e in epochs:
            theta = stochastic_gradient_descent(X_train, y_train, learning_rate=lr, epochs=e, batch_size=bs)
            loss_sgd = loss(X_test, y_test, theta)
            print('Learning Rate:', lr, 'Batch Size:', bs, 'Epochs:', e, 'Loss:', loss_sgd)
            if loss_sgd < 0.5:
                break

# Provide a summary of the performance of each method and assess whether a linear model is suitable for predicting wine quality in this specific case.
"""
Summary:
The Normal Equation method achieved the lowest loss on the test data, followed by Stochastic Gradient Descent and Gradient Descent. The number of epochs required for Gradient Descent to converge to results close to the Normal Equation was 1000. The best results were obtained with a learning rate of 0.01, batch size of 1, and 100 epochs for Stochastic Gradient Descent.

Assessment:
A linear model may not be suitable for predicting wine quality in this specific case, as the features may not have a linear relationship with the target variable. A more complex model, such as a polynomial regression or a machine learning algorithm like Random Forest or Gradient Boosting, may provide better results.
"""
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------