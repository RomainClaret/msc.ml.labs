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

filename = 'adult.test'
file_handler = open(filename, 'r').readlines()[1:]
prefix_file = "adult_2021_cw_"
week_number = 1
split_into = 10
line_count = 0
file_length = len(file_handler)

for i in range(0,file_length):
    if i % ((file_length)//split_into) == 0 and i+((file_length//split_into)//2) < file_length:
        open(str(prefix_file)+str(week_number) + ".csv", "w+").writelines(file_handler[i:i+(file_length//split_into)])
        week_number += 1


# RUN PIPELINE MODEL FROM OTHER FILE
#input file, and save the predictions into a different file.
#Example:
#Let's say you have the input data weekly in the file adult_2021_cw_12.csv.
#This second script should read the input from this file and use the classifier to make predictions and write those predictions in the file adult_2021_cw_12_pred.csv .

# load pipeline model
pipeline_model = pickle.load( open("grid_search_model.pickle", "rb" ))

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
    
    # lab 03 results:
    # adult_2021_cw_1.csv accuracy_score: 0.8293736501079914 
    # adult_2021_cw_2.csv accuracy_score: 0.8503253796095445 
    # adult_2021_cw_3.csv accuracy_score: 0.8427807486631016 
    # adult_2021_cw_4.csv accuracy_score: 0.8307860262008734 
    # adult_2021_cw_5.csv accuracy_score: 0.8507462686567164 
    # adult_2021_cw_6.csv accuracy_score: 0.854978354978355 
    # adult_2021_cw_7.csv accuracy_score: 0.8545454545454545 
    # adult_2021_cw_8.csv accuracy_score: 0.8514531754574811 
    # adult_2021_cw_9.csv accuracy_score: 0.8296943231441049 
    # adult_2021_cw_10.csv accuracy_score: 0.8574537540805223 