#!/usr/bin/env python3

# Master Class: Machine Learning (5MI2018)
# Faculty of Economic Science
# University of Neuchatel (Switzerland)

# Authors: 
# - Romain Claret @RomainClaret
# - Sylvain Robert-Nicoud @Nic0uds

# Here we build and run the KNN classification model

from sklearn.neighbors import KNeighborsClassifier

# kNN exemple
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
def demo():
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    n_neighbors = 3

    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(X, y)

    print(neigh.predict([[1.1]])) #[0]
    print(neigh.predict_proba([[0.9]])) #[[0.66666667 0.33333333]]