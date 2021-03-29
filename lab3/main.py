#!/usr/bin/env python3
# 29.03.19
# Assignment lab 03

# Master Class: Machine Learning (5MI2018)
# Faculty of Economic Science
# University of Neuchatel (Switzerland)
# Lab 3, see ML21_Exercise_3.pdf for more information

# https://github.com/RomainClaret/msc.ml.labs

# Authors: 
# - Romain Claret @RomainClaret
# - Sylvain Robert-Nicoud @Nic0uds

# PART: DATA CLEANING AND PREPARATION
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle


# get the features names and the values of the categories from adult.names (build a dictionary)
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


# get meaningful columns
continuous_features = [k for k, v in data_dict.items() if v == ""]
unencoded_features = ["workclass", "race", "native-country"]
encoded_features = [c for c in df_train if c.startswith('encoded')]
columns = continuous_features+unencoded_features+encoded_features
    

# PART PIPELINE
# 1. Use a Pipeline for the transformation and model training.
# Here we build out pipeline
# First we create a ColumnTransformer for the Categorical (ignoring unknown values) and Numeric features
# Second we pipe the features and the classifier with the parameters from lab2
    
# Standardizing of numeric values
# Doesn't have a lot of meaning in the case of decision trees as it's not using distances (like KNN)
# But it's just a pedagological flavor and to use the ColumnTranformer for whatever reason 
# https://stats.stackexchange.com/questions/10289/whats-the-difference-between-normalization-and-standardization

# We choose a the best parameters from lab2 for the decision tree
# depth=8 Train accuracy_score 0.8550292179535
# depth=8 Test accuracy_score 0.8465108569534229
# depth=8 Evaluation accuracy_score 0.8469455511288181 

feature_transformation = ColumnTransformer(transformers=[
    ('categorical', OneHotEncoder(handle_unknown='ignore'), unencoded_features+encoded_features[:-1]),
    ('numeric', StandardScaler(), continuous_features)
])

adult_pipeline = Pipeline(steps=[
  ('features', feature_transformation),
  ('classifier', DecisionTreeClassifier(criterion='gini', random_state=1, max_depth=8))
])

# building our pipeline-driven model
pipeline_model = adult_pipeline.fit(df_train, df_train["encoded_income"])
    
#remove the fake rows
df_train = df_train[:-len(df_fake)]
df_evaluate = df_evaluate[:-len(df_fake)]    


# make training and testings sets
X_train, X_test, y_train, y_test = train_test_split(df_train,
                                                    df_train["encoded_income"],
                                                    test_size=0.2,
                                                    random_state=1)

# make evaluation sets
X_evaluate = df_evaluate
y_evaluate = df_evaluate["encoded_income"]


# PART EVALUTATION
# evaluate Decision Tree with our pipeline

cl_name = "Evaluate Decision Tree Classifier Pipeline on new data"
print("*"*len(cl_name))
print(cl_name)
print("*"*len(cl_name),'\n')

y_hat_dtree_train = pipeline_model.predict(X_train)
y_hat_dtree_test = pipeline_model.predict(X_test)
y_hat_dtree_evaluate = pipeline_model.predict(X_evaluate)

print("Train accuracy_score",accuracy_score(y_train,y_hat_dtree_train))
print("Test accuracy_score",accuracy_score(y_test,y_hat_dtree_test))
print("Evaluation accuracy_score",accuracy_score(y_evaluate,y_hat_dtree_evaluate),"\n")


# SERALIZE PIPELINE
# 2. Save the Pipeline (using pickle) into a file
pickle.dump(pipeline_model, open("pipeline_model.pickle", "wb" ))


#def main():
#    model_knn.demo()
#    model_decision_tree.demo()
#
#if __name__ == "__main__":
#    main()
