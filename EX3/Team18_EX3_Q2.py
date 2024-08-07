"""Authors: Eliyahu cohen, id 304911084
            Daniel liberman, id 208206946

***both authors have contributed equally to the assignment.***
we were working together on the assignment and we both participated in the writing of the code and the writing of the report
---------------------------------------------------------------------------------
Short Description:
# Given a dataset of an EMG data for gestures, we trained and evaluated 3 different classifiers.
# The classifiers include: Logistic Regression, Nearest Neighbors, and Naive Bayes.
# For each classifier, the accuracy and other metrics were printed, and a graph of the accuracy vs. number of participants was plotted.

# Results
# The results indicate that Naive Bayes and K-Nearest Neighbors (KNN) classifiers perform better compared to Logistic Regression on this dataset.
# Naive Bayes shows the highest accuracy, likely due to its strong assumptions of feature independence, which work well with this dataset.
# KNN also performs well, benefiting from its non-parametric nature, making fewer assumptions about the data.
# Logistic Regression, however, shows lower accuracy, possibly due to the linear decision boundaries it creates, which may not capture the complexities in the data.
# To improve the results, we introduced grid search over the LR and KNN parameters and also employed cross-validation for more robust estimate of calssifiers performance
# Logistic Regression was tuned to find the optimal regularization strength (C) of 0.1, showing a slight improvement in accuracy from 0.2148 to 0.215.
# Nearest Neighbors however, despite the tuning process resulting with K=3, exhibited accuracy dropping from 0.5724 to 0.5637.
# This indicates that the tuning may not align well with the dataset's characteristics or could be influenced by other model or data specifics.
# Naive Bayes, doesn't require hyperparameter tuning like other algorithms because its assumptions are relatively simple and based on the distribution of data (assuming normal distribution for each feature)
# Overall, Naive Bayes and KNN are recommended for this dataset due to their higher accuracy and ability to handle non-linear relationships.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

def load_data():
    """Load training and testing data from pickle files."""
    with open('emg_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('emg_test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    return train_data['features'], train_data['label'], test_data['features'], test_data['label']

def preprocess_data(X_train, X_test):
    """Standardize features by removing the mean and scaling to unit variance."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

def initialize_classifiers(X_train_scaled, y_train):
    """Initialize classifiers and perform grid search to tune parameters where applicable."""
    classifiers = {
        'Logistic Regression': LogisticRegression(multi_class='ovr', max_iter=1000),
        'Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }
    param_grid = {
        'Logistic Regression': {'C': [0.01, 0.1, 1, 10, 100]},
        'Nearest Neighbors': {'n_neighbors': [3, 5, 7, 9]}
    }
    tuned_classifiers = {}
    for name, clf in classifiers.items():
        if name in param_grid:
            grid_search = GridSearchCV(clf, param_grid[name], cv=5, n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            tuned_classifiers[name] = grid_search.best_estimator_
            print(f"Best parameters for {name}: {grid_search.best_params_}")
    return classifiers, tuned_classifiers

def evaluate_accuracy(clf, X, y, X_test, y_test, size, max_participants):
    """Evaluate the accuracy of a classifier given a proportion of the training data."""
    proportion = size / max_participants
    train_size = max(1, min(int(proportion * len(X)), len(X) - 1))
    X_train_subset, _, y_train_subset, _ = train_test_split(X, y, train_size=train_size, random_state=42)
    clf.fit(X_train_subset, y_train_subset)
    return accuracy_score(y_test, clf.predict(X_test))

def calculate_accuracies(classifiers, X, y, X_test, y_test, max_participants):
    """Calculate accuracies for each classifier across varying data proportions using parallel processing."""
    participant_sizes = np.linspace(1, max_participants, num=max_participants, dtype=int)
    results = {}
    for name, clf in classifiers.items():
        accuracies = Parallel(n_jobs=-1)(
            delayed(evaluate_accuracy)(clf, X, y, X_test, y_test, size, max_participants) for size in participant_sizes)
        results[name] = accuracies
        print(f"{name} Accuracy: {accuracies[-1]}")  # Print the last accuracy (full data)
    return results, participant_sizes

def plot_accuracy_vs_participants(classifiers, tuned_classifiers, X, y, X_test, y_test, max_participants):
    """Plot accuracy vs. number of participants for both default and tuned classifiers."""
    results_default, participant_sizes = calculate_accuracies(classifiers, X, y, X_test, y_test, max_participants)
    results_tuned, _ = calculate_accuracies(tuned_classifiers, X, y, X_test, y_test, max_participants)
    plt.figure(figsize=(12, 8))
    for name, accuracies in results_default.items():
        plt.plot(participant_sizes, accuracies, label=f'{name} Default')
    for name, accuracies in results_tuned.items():
        plt.plot(participant_sizes, accuracies, '--', label=f'{name} Tuned')
    plt.xlabel('Number of Participants in Training Data')
    plt.ylabel('Accuracy (%)')
    plt.title('Classifier Accuracy vs Number of Participants (Default vs. Tuned)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    max_participants = 32
    X_train, y_train, X_test, y_test = load_data()
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
    classifiers, tuned_classifiers = initialize_classifiers(X_train_scaled, y_train)
    plot_accuracy_vs_participants(classifiers, {k: v for k, v in tuned_classifiers.items() if k != 'Naive Bayes'}, X_train_scaled, y_train, X_test_scaled, y_test, max_participants)

if __name__ == '__main__':
    main()