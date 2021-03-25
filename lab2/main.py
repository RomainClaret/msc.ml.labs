#!/usr/bin/env python3
# 22.03.19
# Assignment lab 02

# Master Class: Machine Learning (5MI2018)
# Faculty of Economic Science
# University of Neuchatel (Switzerland)
# Lab 2, see ML21_Exercise_1.pdf for more information

# https://github.com/RomainClaret/msc.ml.labs

# Authors: 
# - Romain Claret @RomainClaret
# - Sylvain Robert-Nicoud @Nic0uds

# 1. Convert attributes from nominal to numeric, using different methods.

# PART: DATA CLEANING AND PREPARATION
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# get the features names and the values of the categories from adult.names (features and values linked by the index)
data_dict = {}
with open('adult.names') as f:
    for l in f:
        if l[0] == '|' or ':' not in l: continue
        c = l.split(':')
        if c[1].startswith(' continuous'): data_dict[c[0]] = ""
        else: data_dict[c[0]] = c[1].replace("\n","").replace(".","").replace(" ","").split(",")

# in the specifications (adult.names): Unknown values are replaced with the character '?'
header = list(data_dict.keys())+['income']
df_train = pd.read_table("adult.data", sep=r',\s', na_values='?', header=None, names=header).dropna()
df_evaluate = pd.read_table("adult.test", sep=r',\s', na_values='?', skiprows=[0], header=None, names=header).dropna()


# Standardizing of numeric values
# Doesn't have a lot of meaning in the case of decision trees as it's not using distances (like KNN)
# But it's just a pedagological flavor
# https://stats.stackexchange.com/questions/10289/whats-the-difference-between-normalization-and-standardization
#for c in df_train.select_dtypes(exclude=['object']):
#    df_train["stand_"+c] = df_train[c] - (df_train[c].mean() / df_train[c].std())
#    
#for c in df_evaluate.select_dtypes(exclude=['object']):
#    df_evaluate["stand_"+c] = df_evaluate[c] - (df_evaluate[c].mean() / df_evaluate[c].std())


# droping the education because it's redundant with education-num
# droping the occupation because it's not generic enough, we have much more categories that those captured in the training sample
# droping the relationship because it's not generic enough, we have much more categories that those captured in the training sample
drop_list = ["education", "occupation", "relationship"]
df_train = df_train.drop(columns=drop_list)
df_evaluate = df_evaluate.drop(columns=drop_list)


# reducing categories with multiple options into lower dimensions classification (into binary preferably) when possible
# - marital-status could be reduced as Married or Not-Married
# marital-status ['Never-married' 'Married-civ-spouse' 'Divorced' 'Married-spouse-absent' 'Separated' 'Married-AF-spouse' 'Widowed']
# - workclass could be recuded to 3 dimensions: Government, Private, and Self-Employment
# Note that we take into consideration all the options for the category from the specifications
# ['State-gov' 'Self-emp-not-inc' 'Private' 'Federal-gov' 'Local-gov' 'Self-emp-inc' 'Without-pay']
dict_replace = {
    'marital-status' : {
        'Never-married': 'Not-Married',
        'Married-civ-spouse': 'Married',
        'Divorced': 'Not-Married',
        'Married-spouse-absent': 'Married',
        'Separated': 'Married',
        'Married-AF-spouse': 'Married',
        'Widowed': 'Not-Married'
        },
    'workclass': {
        'State-gov': 'Government',
        'Self-emp-not-inc': 'Self-Employment',
        'Federal-gov': 'Government',
        'Local-gov': 'Government',
        'Self-emp-inc': 'Self-Employment'
        }
}

df_train.replace(dict_replace, inplace=True)
df_evaluate.replace(dict_replace, inplace=True)


# uniformizing the categories between the training and evaluation datasets
# indeed, there is a . at the end of the value in the evaluation dataset for the income category and not in the training dataset
df_evaluate["income"].replace({"<=50K.":"<=50K", ">50K.":">50K"}, inplace=True)


# for binary categories we will be using a label encoder
# - marital-status, sex, income
for l in ["marital-status", "sex", "income"]:
    l_enc = LabelEncoder()
    encoder_train = l_enc.fit(df_train[l])
    encoder_evaluate = l_enc.fit(df_evaluate[l])
    df_train["encoded_"+l] = encoder_train.transform(df_train[l])
    df_evaluate["encoded_"+l] = encoder_evaluate.transform(df_evaluate[l])

    
# For non-binary categories, first we check the specifications of the dataset to validate all the options per category (we have data_dict)
# Indeed, the values in the categories are not always all present in a dataset
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
# and our custom category: workclass: Government, Private, and Self-Employment
# adding temporary fake data for the one hot encoder
fake_row = df_train[:1].copy()
df_fake = pd.DataFrame(data=fake_row, columns=df_train.columns)

cats_nonbinary = ["race", "native-country"]

for c in cats_nonbinary:
    for v in data_dict[c]:
        fake_row[c] = v
        df_fake = df_fake.append(fake_row, ignore_index=True)
        
cat_workclass = ["Government", "Private", "Self-Employment"]
for cw in cat_workclass:
    fake_row["workclass"] = cw
    df_fake = df_fake.append(fake_row, ignore_index=True)
    
df_train = df_train.append(df_fake).reset_index(drop=True)
df_evaluate = df_evaluate.append(df_fake).reset_index(drop=True)
    
    
# for non-binary categories we will be using a onehot encoder as decision trees are sensitive to leaves values
# note that get_dummies from pandas is exactly doing this without the complexity of using OneHotEncoder manually from sklearn
# - workclass, race, native-country
for l in ["workclass", "race", "native-country"]:
    df_train=pd.concat([df_train, pd.get_dummies(df_train[l], prefix="encoded_"+l)], axis=1)
    df_evaluate=pd.concat([df_evaluate, pd.get_dummies(df_evaluate[l], prefix="encoded_"+l)], axis=1)

    
#remove the fake rows
df_train = df_train[:-len(df_fake)]
df_evaluate = df_evaluate[:-len(df_fake)]    

# keep meaningful columns
continuous_features = [k for k, v in data_dict.items() if v == ""]
encoded_features = [c for c in df_train if c.startswith('encoded')]
columns = continuous_features+encoded_features
columns.remove("encoded_income")    

# make training and testings sets
X_train, X_test, y_train, y_test = train_test_split(df_train[columns],
                                                    df_train["encoded_income"],
                                                    test_size=0.2,
                                                    random_state=1)

# make evaluation sets
X_evaluate = df_evaluate[columns]
y_evaluate = df_evaluate["encoded_income"]


# PART CLASSIFICATIONS
# We used the training set to train our models and then we tested them on the testing set
# We made a loop to try a parameter (depth for decision tree, and k for KNN) and select the best one based on the testing set.

# run a Decision Tree Classifier
cl_name = "Searching best Depth for Decision Tree Classifier"
print("*"*len(cl_name))
print(cl_name)
print("*"*len(cl_name),'\n')

parameter_dtree_min = 1
parameter_dtree_max = 15
preds_dtree_train=[]
preds_dtree_test=[]
for depth in range(parameter_dtree_min,parameter_dtree_max):
    cl_dtree = DecisionTreeClassifier(criterion='gini', random_state=1, max_depth=depth)
    dtree_model = cl_dtree.fit(X_train, y_train)
    y_hat_dtree_train = dtree_model.predict(X_train)
    y_hat_dtree_test = dtree_model.predict(X_test)
    preds_dtree_train.append(accuracy_score(y_train, y_hat_dtree_train))
    preds_dtree_test.append(accuracy_score(y_test, y_hat_dtree_test))
    #print(depth,"Train accuracy_score",preds_train[-1])
    #print(depth,"Test accuracy_score",preds_test[-1],"\n")
    
plt.scatter(range(parameter_dtree_min, parameter_dtree_max), preds_dtree_train, c="b", label="train score")
plt.scatter(range(parameter_dtree_min, parameter_dtree_max), preds_dtree_test, c="r", label="test score")
plt.legend(loc="upper left")
plt.title('DecisionTreeClassifier: accuracy_score vs depth')
plt.xlabel('depth')
plt.ylabel('accuracy_score')
plt.savefig('DecisionTreeClassifier.png')
#plt.show()
plt.clf()


# PART EVALUTATION
# We evaluate the best parameter found previously with the evaluation set (adult.test)

#evaluate Decision Tree on new data
#present depth with best score for evaluation dataset
cl_name = "Evaluate Decision Tree Classifier on new data"
print("*"*len(cl_name))
print(cl_name)
print("*"*len(cl_name),'\n')
max_dtree_index = preds_dtree_test.index(max(preds_dtree_test))
best_depth = list(range(parameter_dtree_min, parameter_dtree_max))[max_dtree_index]
cl_dtree = DecisionTreeClassifier(criterion='gini', random_state=1, max_depth=best_depth)
dtree_model = cl_dtree.fit(X_train, y_train)
y_hat_dtree_train = dtree_model.predict(X_train)
y_hat_dtree_test = dtree_model.predict(X_test)
y_hat_dtree_evaluate = dtree_model.predict(X_evaluate)
print("depth="+str(best_depth),"Train accuracy_score",accuracy_score(y_train, y_hat_dtree_train))
print("depth="+str(best_depth),"Test accuracy_score",accuracy_score(y_test, y_hat_dtree_test))
print("depth="+str(best_depth),"Evaluation accuracy_score",accuracy_score(y_evaluate, y_hat_dtree_evaluate),"\n")


plt.close()


#def main():
#    model_knn.demo()
#    model_decision_tree.demo()
#
#if __name__ == "__main__":
#    main()
