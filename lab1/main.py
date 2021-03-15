#!/usr/bin/env python3

# Master Class: Machine Learning (5MI2018)
# Faculty of Economic Science
# University of Neuchatel (Switzerland)
# Lab 1, see assignment_lab_01.pdf for more information

# Authors: 
# - Romain Claret @RomainClaret
# - Sylvain Robert-Nicoud @Nic0uds

# PART: DATA CLEANING AND PREPARATION

# in the specifications (adult.names): Unknown values are replaced with the character '?'
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def prepare_datasets():
    header = ['age','workclass','fnlwgt','education','education-num',
              'marital-status','occupation','relationship','race','sex',
              'capital-gain','capital-loss','hours-per-week','native-country','income']

    train = pd.read_table("adult.data", sep=r',\s', na_values='?', header=None, names=header).dropna()
    evaluate = pd.read_table("adult.test", sep=r',\s', na_values='?', skiprows=[0], header=None, names=header).dropna()
    return train, evaluate

df_train, df_evaluate = prepare_datasets()

# only keep non-categorical values but keeping the income catagory
# careful with the income category not the same in train and test sets
df_train = df_train.select_dtypes(exclude=['object']).join(df_train.income.replace("<=50K",0).replace(">50K",1))
df_evaluate = df_evaluate.select_dtypes(exclude=['object']).join(df_evaluate.income.replace("<=50K.",0).replace(">50K.",1))

# make training and testings sets
X_train, X_test, y_train, y_test = train_test_split(
    df_train[list(df_train.columns)[:-1]],
    df_train[list(df_train.columns)[-1]],
    test_size=0.2,
    random_state=42)

# make evaluation sets
X_evaluate = df_evaluate[list(df_evaluate.columns)[:-1]]
y_evaluate = df_evaluate[list(df_evaluate.columns)[-1]]


# PART CLASSIFICATIONS

# run a Decision Tree Classifier
cl_name = "Decision Tree Classifier"
print("*"*len(cl_name))
print(cl_name)
print("*"*len(cl_name),'\n')

parameter_dtree_min = 1
parameter_dtree_max = 15
preds_dtree_train=[]
preds_dtree_test=[]
for depth in range(parameter_dtree_min,parameter_dtree_max):
    cl_dtree = DecisionTreeClassifier(criterion='gini', random_state=1,max_depth=depth)
    dtree_model = cl_dtree.fit(X_train,y_train)
    y_hat_dtree_train = dtree_model.predict(X_train)
    y_hat_dtree_test = dtree_model.predict(X_test)
    preds_dtree_train.append(accuracy_score(y_train,y_hat_dtree_train))
    preds_dtree_test.append(accuracy_score(y_test,y_hat_dtree_test))
    #print(depth,"Train accuracy_score",preds_train[-1])
    #print(depth,"Test accuracy_score",preds_test[-1],"\n")
    
plt.scatter(range(parameter_dtree_min,parameter_dtree_max),preds_dtree_train,c="b",label="train score")
plt.scatter(range(parameter_dtree_min,parameter_dtree_max),preds_dtree_test,c="r",label="test score")
plt.legend(loc="upper left")
plt.title('DecisionTreeClassifier: accuracy_score vs depth')
plt.xlabel('depth')
plt.ylabel('accuracy_score')
plt.savefig('DecisionTreeClassifier.png')
#plt.show()
plt.clf()

#present depth with best score for evaluation dataset
max_dtree_index = preds_dtree_test.index(max(preds_dtree_test))
best_depth = list(range(parameter_dtree_min,parameter_dtree_max))[max_dtree_index]
cl_dtree = DecisionTreeClassifier(criterion='gini', random_state=1,max_depth=best_depth)
dtree_model = cl_dtree.fit(X_train,y_train)
y_hat_dtree_train = dtree_model.predict(X_train)
y_hat_dtree_test = dtree_model.predict(X_test)
y_hat_dtree_evaluate = dtree_model.predict(X_evaluate)
print("depth="+str(best_depth),"Train accuracy_score",accuracy_score(y_train,y_hat_dtree_train))
print("depth="+str(best_depth),"Test accuracy_score",accuracy_score(y_test,y_hat_dtree_test))
print("depth="+str(best_depth),"Evaluation accuracy_score",accuracy_score(y_evaluate,y_hat_dtree_evaluate),"\n")
    

# run a KNN Classifier
cl_name = "K-Nearest Neighbors Classifier"
print("*"*len(cl_name))
print(cl_name)
print("*"*len(cl_name),'\n')

parameter_knn_min = 2
parameter_knn_max = 30
preds_knn_train=[]
preds_knn_test=[]
for k in range(parameter_knn_min,parameter_knn_max):
    cl_knn = KNeighborsClassifier(n_neighbors = k)
    knn_model = cl_knn.fit(X_train, y_train)
    y_hat_dtree_train = knn_model.predict(X_train)
    y_hat_dtree_test = knn_model.predict(X_test)
    preds_knn_train.append(accuracy_score(y_train,y_hat_dtree_train))
    preds_knn_test.append(accuracy_score(y_test,y_hat_dtree_test))
    #print(k,"Train accuracy_score",preds_train[-1])
    #print(k,"Test accuracy_score",preds_test[-1],"\n")

plt.scatter(range(parameter_knn_min,parameter_knn_max),preds_knn_train,c="b",label="train score")
plt.scatter(range(parameter_knn_min,parameter_knn_max),preds_knn_test,c="r",label="test score")
plt.legend(loc="upper left")
plt.title('KNeighborsClassifier: accuracy_score vs k neighbors')
plt.xlabel('k neighbors')
plt.ylabel('accuracy_score')
plt.savefig('KNeighborsClassifier.png')
#plt.show()
plt.clf()

#present k with best score for evaluation dataset
max_knn_index = preds_knn_test.index(max(preds_knn_test))
best_k = list(range(parameter_knn_min,parameter_knn_max))[max_knn_index]
cl_knn = KNeighborsClassifier(n_neighbors = best_k)
knn_model = cl_knn.fit(X_train,y_train)
y_hat_dtree_train = knn_model.predict(X_train)
y_hat_dtree_test = knn_model.predict(X_test)
y_hat_dtree_evaluate = knn_model.predict(X_evaluate)
print("k="+str(best_k),"Train accuracy_score",accuracy_score(y_train,y_hat_dtree_train))
print("k="+str(best_k),"Test accuracy_score",accuracy_score(y_test,y_hat_dtree_test))
print("k="+str(best_k),"Evaluation accuracy_score",accuracy_score(y_evaluate,y_hat_dtree_evaluate),"\n")


#def main():
#    model_knn.demo()
#    model_decision_tree.demo()
#
#if __name__ == "__main__":
#    main()
