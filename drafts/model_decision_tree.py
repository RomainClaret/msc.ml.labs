#!/usr/bin/env python3

# Master Class: Machine Learning (5MI2018)
# Faculty of Economic Science
# University of Neuchatel (Switzerland)
# Lab 1, see assignment_lab_01.pdf for more information

# Authors: 
# - Romain Claret @RomainClaret
# - Sylvain Robert-Nicoud @Nic0uds

# Here we build and run the Decision Tree classification model

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

# Decision tree example
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
def demo():
    clf = DecisionTreeClassifier(random_state=0)
    iris = load_iris()
    print(cross_val_score(clf, iris.data, iris.target, cv=10))
    # array([1.        , 0.93333333, 1.        , 0.93333333, 0.93333333,
    #       0.86666667, 0.93333333, 1.        , 1.        , 1.        ])