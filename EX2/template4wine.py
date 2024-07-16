import numpy as np 
from sklearn.model_selection import train_test_split
import pickle

data = pickle.load(file=open('wine_red_dataset.pkl', "rb"))
X = data['features'] # ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
Y = data['quality'] # [Quality]
K = data['feature_names'] # Strings of feature names

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
