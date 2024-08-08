"""Authors: Eliyahu cohen, id 304911084
            Daniel liberman, id 208206946

***both authors have contributed equally to the assignment.***
we were working together on the assignment and we both participated in the writing of the code and the writing of the report
---------------------------------------------------------------------------------
Short Description:

#Given a dataset of an occupancy detection system we trained and evaluated 5 different classifiers.
#The classifiers include: Logistic Regression, Nearest Neighbors, Linear SVM, Gaussian SVM, and Naive Bayes.
#For each classifier, the accuracy was printed and a graph of the accuracy vs. number of test samples was plotted.

#Results:
# The results indicate that SVM algorithms (both Linear and RBF) provide the best accuracy for this dataset.
# SVMs are particularly effective for complex classification problems because they can create non-linear decision boundaries (especially RBF SVM).
# Logistic Regression, while simpler, also performs very well, close to SVM.
# K-Nearest Neighbors (KNN) shows slightly lower accuracy and tends to be sensitive to the choice of k and the number of samples.
# we found that K=50 provides the best accuracy of 97.37
# Naive Bayes, assuming feature independence, performs well with a large dataset but slightly lower than SVM and Logistic Regression.

# Overall, SVM algorithms are recommended for their high accuracy and ability to handle high-dimensional spaces effectively.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

names = ['Logistic Regression','Nearest Neighbors',
'Linear SVM',
'RBF SVM',
'Gaussian Naive-Bayes']

classifiers = [LogisticRegression(solver='liblinear', random_state=0,C=10),
KNeighborsClassifier(n_neighbors=50),
SVC(kernel="linear", C=0.5, probability = True),
SVC(gamma=0.8, C=0.5, probability = True), # RBF is the default
GaussianNB()]

#make a dictionary of the classifiers
classifiers = {names[i]: classifiers[i] for i in range(len(names))}
def accuracy_of_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# plot the learning curves for the classifiers
def plot_learning_curve(classifiers, X_train, y_train, X_test, y_test):
    train_sizes = np.linspace(0.1, 0.99, 5)
    plt.figure(figsize=(12, 8))
    for name, clf in classifiers.items():
        test_scores = []
        for size in train_sizes:
            X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)
            clf.fit(X_train_subset, y_train_subset)
            test_scores.append(clf.score(X_test, y_test))
            y_pred = clf.predict(X_test)
        print(f'{name}: {accuracy_score(y_test, y_pred) * 100:.2f}%')
        plt.plot(train_sizes * len(X_train), test_scores, label=f'{name}')

    plt.xlabel('Number of samples',fontweight='bold')
    plt.ylabel('Success rate (%)',fontweight='bold')
    plt.title('Learning Curves for Classifiers', fontsize=15, fontweight='bold')
    plt.legend(fontsize=10, loc='lower right')
    plt.show()


def main():
    print("Loading data...\n")
    print('#'*100)
    plot_learning_curve(classifiers, X_train_scaled, y_train, X_test_scaled, y_test) # plot the learning curves for the classifiers
    print('#'*100)

if __name__ == '__main__':
    main()
