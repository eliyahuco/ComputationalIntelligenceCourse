#Authors: Eliyahu cohen, id 304911084, Daniel liberman, id 208206946
#both authors have contributed equally to the assignment.
#we were working together on the assignment and we both participated in the writing of the code and the writing of the report

#Short description:
#Given a dataset of an occupancy detection system we trained and evaluated 5 different classifiers.
#The classifiers include: Logistic Regression, Nearest Neighbors, Linear SVM, RBF SVM and Naive Bayes.
#For each classifier, the accuracy was printed and a graph of the accuracy vs. number of test sumples was plotted.

#Results:
# The results indicate that SVM algorithms (both Linear and RBF) provide the best accuracy for this dataset.
# SVMs are particularly effective for complex classification problems because they can create non-linear decision boundaries (especially RBF SVM).
# Logistic Regression, while simpler, also performs very well, close to SVM.
# K-Nearest Neighbors (KNN) shows slightly lower accuracy and tends to be sensitive to the choice of k and the number of samples.
# Naive Bayes, assuming feature independence, performs well with a large dataset but slightly lower than SVM and Logistic Regression.
# To improve the results, we introduced grid search over the parameters of the different classifiers
# The KNN classifier was tuned with K=10 showing slight accuracy improvment from 97.22% to 97.34%
# As for the the other classifiers, while performing slightly better/worse for lower number of samples, eventualy all converge to the same pre-tuned results.
# This could be partialy explained due to the initially high accuracies of the pre-tuned classifiers.
# Overall, SVM algorithms are recommended for their high accuracy and ability to handle high-dimensional spaces effectively.

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

# Load the data
with open('occupancy_train.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('occupancy_test.pkl', 'rb') as f:
    test_data = pickle.load(f)

X_train, y_train = train_data['features'], train_data['label']
X_test, y_test = test_data['features'], test_data['label']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameters for tuning
param_grid = {
    'Logistic Regression': {'C': [0.1, 1, 10, 100]},
    'Nearest Neighbors': {'n_neighbors': [3, 5, 7, 10]},
    'Linear SVM': {'C': [0.1, 1, 10]},
    'RBF SVM': {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
}

# Initialize classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Nearest Neighbors': KNeighborsClassifier(),
    'Linear SVM': SVC(kernel='linear'),
    'RBF SVM': SVC(kernel='rbf'),
    'Naive Bayes': GaussianNB()
}

untuned_accuracies = {}

# Train and evaluate classifiers before tuning
for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    untuned_accuracies[name] = accuracy
    print(f"{name} Untuned Accuracy: {accuracy * 100:.2f}%")

tuned_classifiers = {}

# Tune and evaluate classifiers
for name, clf in classifiers.items():
    if name in param_grid:
        grid_search = GridSearchCV(clf, param_grid[name], cv=5)
        grid_search.fit(X_train_scaled, y_train)
        best_clf = grid_search.best_estimator_
        best_clf.fit(X_train_scaled, y_train)
        y_pred = best_clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        tuned_classifiers[name] = best_clf
        print(f"{name} Tuned Accuracy: {accuracy * 100:.2f}%")
        print(f"Best parameters for {name}: {grid_search.best_params_}")

# Function to plot the learning curves for both tuned and untuned classifiers
def plot_learning_curve(classifiers, tuned_classifiers, X, y, X_test, y_test):
    train_sizes = np.linspace(0.1, 0.99, 5)
    plt.figure(figsize=(12, 8))
    for name, clf in classifiers.items():
        test_scores = []
        for size in train_sizes:
            X_train_subset, _, y_train_subset, _ = train_test_split(X, y, train_size=size, random_state=42)
            clf.fit(X_train_subset, y_train_subset)
            test_scores.append(clf.score(X_test, y_test))
        plt.plot(train_sizes * len(X), test_scores, label=f'{name} Default')

    for name, clf in tuned_classifiers.items():
        test_scores = []
        for size in train_sizes:
            X_train_subset, _, y_train_subset, _ = train_test_split(X, y, train_size=size, random_state=42)
            clf.fit(X_train_subset, y_train_subset)
            test_scores.append(clf.score(X_test, y_test))
        plt.plot(train_sizes * len(X), test_scores, '--', label=f'{name} Tuned')

    plt.xlabel('Number of samples')
    plt.ylabel('Accuracy (%)')
    plt.title('Learning Curves for Classifiers')
    plt.legend(fontsize=10, loc='upper right')
    plt.show()

# Plot the learning curves for both tuned and untuned classifiers
plot_learning_curve(classifiers, tuned_classifiers, X_train_scaled, y_train, X_test_scaled, y_test)

