#!/usr/bin/env python3
# 12.04.21
# Assignment lab 04

# Master Class: Machine Learning (5MI2018)
# Faculty of Economic Science
# University of Neuchatel (Switzerland)
# Lab 4, see ML21_Exercise_4.pdf for more information

# https://github.com/RomainClaret/msc.ml.labs

# Authors: 
# - Romain Claret @RomainClaret
# - Sylvain Robert-Nicoud @Nic0uds

import warnings
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")


# SPLITING ADULT.TEST FILE IN SUBFILES
#spliting the adult.test file into several files to simulate weeks

#filename = 'adult.test'
#file_handler = open(filename, 'r').readlines()
#prefix_file = "adult_2021_cw_"
#week_number = 1
#split_into = 10
#line_count = 0
#with open(filename) as f: line_count = sum(1 for _ in f)
#
#for i in range(len(file_handler)):
#    if i % (line_count//split_into) == 0:
#        open(str(prefix_file)+str(week_number) + ".csv", "w+").writelines(file_handler[i:i+1000])
#        week_number += 1


# RUN PIPELINE MODEL FROM OTHER FILE
# 3. Create a second script that will load the Pipeline and use it to predict values from an
#input file, and save the predictions into a different file.
#Example:
#Let's say you have the input data weekly in the file adult_2021_cw_12.csv.
#This second script should read the input from this file and use the classifier to make predictions and write those predictions in the file adult_2021_cw_12_pred.csv .

# load pipeline model
pipeline_model = pickle.load( open("pipeline_model.pickle", "rb" ))

weeks_count = 10
filename = 'adult.test'
prefix_file = "adult_2021_cw_"

# get the features names and the values of the categories from adult.names (build a dictionary)
data_dict = {}
with open('adult.names') as f:
    for l in f:
        if l[0] == '|' or ':' not in l: continue
        c = l.split(':')
        if c[1].startswith(' continuous'): data_dict[c[0]] = ""
        else: data_dict[c[0]] = c[1].replace("\n","").replace(".","").replace(" ","").split(",")
            
header = list(data_dict.keys())+['income']

# for each week based on a count and a naming convention
for i in range (weeks_count):
    filename = str(prefix_file)+str(i+1)+".csv"
    df_weekly = pd.read_table(filename, sep=r',\s', na_values='?', skiprows=[0], header=None, names=header).dropna()
    
    drop_list = ["education", "occupation", "relationship"]
    df_weekly = df_weekly.drop(columns=drop_list)
    
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

    df_weekly.replace(dict_replace, inplace=True)
    
    df_weekly["income"].replace({"<=50K.": "<=50K", ">50K.": ">50K"}, inplace=True)
    
    for l in ["marital-status", "sex", "income"]:
        l_enc = LabelEncoder()
        encoder_weekly = l_enc.fit(df_weekly[l])
        df_weekly["encoded_"+l] = encoder_weekly.transform(df_weekly[l])
    
    y_hat_dtree_weekly = pipeline_model.predict(df_weekly)
    
    pref_filename = str(prefix_file)+str(i+1)+"_pred.csv"
    print(pref_filename, "accuracy_score:",accuracy_score(df_weekly["encoded_income"],y_hat_dtree_weekly),"\n")
    
    # save the prediction into file
    pd.DataFrame(y_hat_dtree_weekly).to_csv(str(pref_filename),header=["pred_income"], index=None)