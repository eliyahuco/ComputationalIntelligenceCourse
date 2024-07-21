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

the sammary of the results is at the end of the code.
---------------------------------------------------------------------------------
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
import time


# Load the dataset
data = pickle.load(open('wine_red_dataset.pkl', "rb"))
X = data['features']  # Features
Y = data['quality']   # Target variable
K = data['feature_names']  # Feature names

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Define the loss function using mean squared error
def Loss(y, y_pred):
    return mean_squared_error(y, y_pred)

# Define the gradient of the loss function
def dLoss_dW(X, y, y_pred):
    M = len(y)
    return -(2 / M) * X.T @ (y - y_pred)

# Define the predict function
def predict(W, X):
    return X @ W

# Define the normal equation function
def normal_equation(X_train, y_train):
    W = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
    return W

# Define the gradient descent function
def gradient_descent(X_train, y_train, X_test, y_test, learning_rate=0.0001, epochs=1000, loss_threshold=0.001, pause=50):
    W = np.ones(X_train.shape[1]) / 10000  # Initial weight vector
    L_train, L_test = [], []
    best_weights, best_loss = None, float('inf')
    best_epoch, no_improvement = 0, 0
    time_start = time.time()

    for epoch in range(epochs):
        y_pred_train = predict(W, X_train)
        W -= learning_rate * dLoss_dW(X_train, y_train, y_pred_train)  # Update weights
        train_loss = Loss(y_train, y_pred_train)
        test_loss = Loss(y_test, predict(W, X_test))
        L_train.append(train_loss)
        L_test.append(test_loss)

        if train_loss + test_loss < best_loss:
            best_loss = train_loss + test_loss
            best_weights = W.copy()
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= pause:
                print(f'Early stopping at epoch {epoch}')
                break

        if test_loss < loss_threshold and epoch > 100:
            break
    time_end = time.time()
    print(f'Time taken for gradient descent: {time_end - time_start:.2f} seconds')
    # Plot the loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(L_train)), L_train, label='Train loss')
    plt.plot(range(len(L_test)), L_test, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Gradient Descent')
    plt.legend()
    plt.show()

    return best_weights, L_test[best_epoch], L_train[best_epoch], best_epoch

# Define the get_batch function
def get_batch(X, y, batch_size=500):
    ix = np.random.choice(X.shape[0], batch_size, replace=False)
    return X[ix, :], y[ix]

# Define the stochastic gradient descent function
def stochastic_gradient_descent(X_train, y_train, X_test, y_test, learning_rate=0.0001, epochs=1000, batch_size=500, loss_threshold=0.001, pause=50):
    W = np.ones(X_train.shape[1]) / 10000  # Initial weight vector
    L_train, L_test = [], []
    best_weights, best_loss = None, float('inf')
    best_epoch, no_improvement = 0, 0
    time_start = time.time()

    for epoch in range(epochs):
        X_batch, y_batch = get_batch(X_train, y_train, batch_size)
        y_pred_batch = predict(W, X_batch)
        W -= learning_rate * dLoss_dW(X_batch, y_batch, y_pred_batch)  # Update weights
        train_loss = Loss(y_batch, y_pred_batch)
        test_loss = Loss(y_test, predict(W, X_test))
        L_train.append(train_loss)
        L_test.append(test_loss)
        if train_loss + test_loss < best_loss:
            best_loss = train_loss + test_loss
            best_weights = W.copy()
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= pause and epoch > 300:
                print(f'Early stopping at epoch {epoch}')
                break

        if test_loss < loss_threshold and epoch > 300:
            break
    time_end = time.time()
    print(f'Time taken for stochastic gradient descent: {time_end - time_start:.2f} seconds')
    # Plot the loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(L_train)), L_train, label='Train loss')
    plt.plot(range(len(L_test)), L_test, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Stochastic Gradient Descent')
    plt.legend()
    plt.show()

    return best_weights, L_test[best_epoch], L_train[best_epoch], best_epoch

# Main function
def main():
    # Normal equation
    learning_rate_gd = 0.00025
    learning_rate_sgd = 0.0001
    W_normal = normal_equation(X_train, y_train)
    pred_quality = predict(W_normal, X_test)
    NEL = Loss(y_test, pred_quality)
    batch_size = 750
    print('#' * 100)
    print('\nNormal equation:')
    print('The loss on the test data using the normal equation is:', NEL)
    print('\nThe weights using the normal equation are:', W_normal)

    # Gradient descent
    print('\n' + '#' * 100)
    print('\ngradient descent:')
    W_gd, loss_test_gd, loss_train_gd, epoch_gd = gradient_descent(X_train, y_train, X_test, y_test, learning_rate=learning_rate_gd, epochs=15000, loss_threshold=NEL * 1.2, pause=150)
    print('\nThe loss on the test data using the gradient descent method is:', loss_test_gd)
    print('The loss on the training data using the gradient descent method is:', loss_train_gd)
    print('Best epoch for gradient descent:', epoch_gd)

    # Stochastic gradient descent
    print('\n' + '#' * 100)
    print('\nstochastic gradient descent:')
    W_sgd, loss_test_sgd, loss_train_sgd, epoch_sgd = stochastic_gradient_descent(X_train, y_train, X_test, y_test, learning_rate=learning_rate_sgd, epochs=15000, batch_size=batch_size, loss_threshold=NEL, pause=150)
    print('\nThe loss on the test data using the stochastic gradient descent method is:', loss_test_sgd)
    print('The loss on the training data using the stochastic gradient descent method is:', loss_train_sgd)
    print('Best epoch for stochastic gradient descent:', epoch_sgd)

    print('\n' + '#' * 100)
    print('Summary:')
    print('Normal Equation test Loss:', NEL)
    print('Gradient Descent test Loss:', loss_test_gd)
    print('Stochastic Gradient Descent test Loss:', loss_test_sgd)
    print(f'\nthe learning rate for gradient descent was {learning_rate_gd} and for stochastic gradient descent was {learning_rate_sgd}')
    print(f'learning rate above 0.00027 make the model diverge')
    print(f'the batch size for stochastic gradient descent was {batch_size}')
    print('\nGradient Descent took', epoch_gd, 'epochs to converge to results close to the Normal Equation')
    print('Stochastic Gradient Descent took', epoch_sgd, 'epochs to converge to results close to the Normal Equation')
    print(f'we found the parameters above as the parameters that gave us the results closest to the normal equation')
    print(f'checking if linear model is suitable for predicting wine quality in this specific case')

    print('\nR2 score for Normal Equation:', r2_score(y_test, pred_quality))
    print('R2 score for Gradient Descent:', r2_score(y_test, predict(W_gd, X_test)))
    print('R2 score for Stochastic Gradient Descent:', r2_score(y_test, predict(W_sgd, X_test)))
    print(f'\nbased on the R2 score and the loss, the linear regression model is not the best choice for this dataset')


    print('\n' + '#' * 100)
    print('the code is finished')
    print('thank you and have a nice day')

if __name__ == '__main__':
    main()

# Summary:
# Normal Equation get the best results with the lowest loss on the test data.
# Gradient Descent took around 13000 epochs to converge to results close to the Normal Equation.
# Stochastic Gradient Descent took around 1000 epochs to converge to results close to the Normal Equation.
# The learning rate for gradient descent was 0.00025 and for stochastic gradient descent was 0.0001.
# the learning rate above 0.00027 make the model diverge.
# learning rate for the gradient descent was chosen to set balance between speed of convergence and accuracy.
# learning rate for the stochastic gradient descent was chosen much lower to avoid divergence, and because the stochastic gradient descent converges faster, we can afford to use a lower learning rate.
# the batch size for stochastic gradient descent was 750, it gave the balance between speed of convergence and accuracy.
# we found the parameters above as the parameters that gave us the results closest to the normal equation.
# based on the R2 score and the loss, we think that the linear regression model is not the best choice for this dataset.






