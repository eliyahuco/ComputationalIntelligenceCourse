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
# The results indicate that Logistic Regression provides the worst accuracy for this dataset.
# Nearest Neighbors shows the best accuracy for this dataset.
# Naive Bayes shows a moderate accuracy for this dataset, but not so far from Nearest Neighbors.
# The accuracy of the classifiers increases as the number of participants increases.
# we try to change the parameters of the classifiers to improve the accuracy, and what we got was the best we could get.
# we used k = 75 for the KNN classifier, and we used C = 0.01 for the Logistic Regression classifier.
# we also used 'lbfgs' as the solver for the Logistic Regression classifier. because it is a solver that is suitable for large datasets.
# we used 'kd_tree' as the algorithm for the KNN classifier. because it is a suitable algorithm for large datasets.

"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the data
with open('emg_train.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('emg_test.pkl', 'rb') as f:
    test_data = pickle.load(f)

X_train, y_train = train_data['features'], train_data['label']
X_test, y_test = test_data['features'], test_data['label']
users_train = train_data['users_train']

# Standardize the features (fit once, transform multiple times)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the classifiers
names = ['Logistic Regression', 'Nearest Neighbors', 'Gaussian Naive-Bayes']
classifiers = [
    LogisticRegression(solver='lbfgs', random_state=0, C=0.01, class_weight='balanced', tol=0.1),
    KNeighborsClassifier(n_neighbors=75, algorithm='kd_tree'),  # Reduced n_neighbors for faster performance
    GaussianNB()
]

# Function to evaluate a single classifier
def evaluate_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

# Precompute user indices for efficiency
user_indices_list = [[j for j, user in enumerate(users_train) if user <= i] for i in range(1, 33)]

def main():
    # Evaluate classifiers on the entire dataset
    for clf in classifiers:
        acc = evaluate_classifier(clf, X_train_scaled, y_train, X_test_scaled, y_test)
        print(f'{clf.__class__.__name__}: {acc:.2%}')

    print('\n')
    print('#'*100)
    print('Prepare the data for the plot')
    print('It might take some time')
    print('#'*100)

    # Evaluate by user count sequentially to reduce overhead
    accs = []
    for user_indices in user_indices_list:
        X_train_subset = X_train_scaled[user_indices]
        y_train_subset = y_train[user_indices]
        acc = [evaluate_classifier(clf, X_train_subset, y_train_subset, X_test_scaled, y_test) for clf in classifiers]
        accs.append(acc)

    # Plot the results
    plt.figure(figsize=(12, 8))
    for i, name in enumerate(names):
        plt.plot(range(1, 33), [acc[i] for acc in accs], label=name)

    plt.title('Accuracy vs. Number of Users', fontsize=15, fontweight='bold')
    plt.xlabel('Number of Users', fontweight='bold')
    plt.ylabel('Success rate (%)', fontweight='bold')
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
