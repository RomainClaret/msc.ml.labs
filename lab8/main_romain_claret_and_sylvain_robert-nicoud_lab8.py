#!/usr/bin/env python3
# 10.05.21
# Assignment lab 08

# Master Class: Machine Learning (5MI2018)
# Faculty of Economic Science
# University of Neuchatel (Switzerland)
# Lab 8, see ML21_Exercise_8.pdf for more information

# https://github.com/RomainClaret/msc.ml.labs

# Authors: 
# - Romain Claret @RomainClaret
# - Sylvain Robert-Nicoud @Nic0uds


# The goal for this exercise is to predict the national based real GDP (rgdpna) OR national based TFP (rtfpna) of each country based on other attributes and the values from previous year(s).


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from joblib import dump

# Import the data and filter the two datasets for the chosen two years.
df = pd.read_stata('pwt100.dta')

# 2. Select the input features that you deem useful for the purpose.
features = [
   "year",
    "country",
    "hc",
    "ctfp",
    "cwtfp",
    "delta",
    "pl_con",
    "pl_da",
    "pl_gdpo",
    "csh_g",
    "pl_c",
    "pl_i",
    "pl_g",
    #"pl_k",
]
features_logarithmic = [
    #"rgdpe",
    "pop",
    "ccon",
    #"rgdpna",
    "rconna",
    "xr",
]

# 1. Select the target variable rgdpna or rtfpna.
targeted_features = [
    "rgdpna", # log required
    #"rtfpna", # log not required
]

# Define the time window
window_frame = 2

# Define the features to apply the time window
features_framed = ["hc","ccon"]

# Apply the logarithmic function on badly proportionated features 
log_transformer = FunctionTransformer(np.log1p)

df_log_features = df[features_logarithmic+targeted_features]
df_log = log_transformer.transform(df_log_features)

# Concat logarithmic features with unlogarithmic features
df_concat = pd.concat([df[features], df_log], axis=1, join="inner")

# Drop rows with na values
df_cleaned = df_concat.dropna()

# normalization of logarithmic features
df_normalized = pd.DataFrame(Normalizer().fit_transform(df_cleaned[features[2:]+features_logarithmic+targeted_features]),columns=features[2:]+features_logarithmic+targeted_features)

# merge logarithmic and non-logarithmic features
df_normalized = pd.concat([df_cleaned[features[:2]].reset_index(drop=True),df_normalized], axis=1)


# Building empty dataframe for time series
tmp_df_windowed = pd.DataFrame(columns=features+features_logarithmic+targeted_features)

# Adding the windowed features columns to empy dataframe
for ff in features_framed:
    for i in range(window_frame):
        tmp_df_windowed[ff+"-"+str(i+1)] = []

# Stack windowed features columns by countries
for c in df_normalized["country"].unique():
    tmp_df_country = df_normalized[df_normalized["country"]==c].sort_values("year")
    for w in range(1,window_frame+1):
        tmp_df_country_framed = tmp_df_country[features_framed][:-w]
        tmp_df_country_framed.index = tmp_df_country_framed.index + w
        for ff in features_framed:
            tmp_df_country_framed = tmp_df_country_framed.rename(columns={ff: ff+"-"+str(w)})

        tmp_df_country = tmp_df_country.join(tmp_df_country_framed)
    tmp_df_windowed = pd.concat([tmp_df_windowed,tmp_df_country])

# Clean the windowed dataframe and reset index
df_normalized_windowed = tmp_df_windowed[tmp_df_windowed.columns.difference(["country","year"])].dropna().reset_index(drop=True)

# build train and target datasets
X = df_normalized_windowed[df_normalized_windowed.columns.difference(targeted_features)]
y = df_normalized_windowed[targeted_features]

# split dataset training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1, test_size=0.2)

# 3. Prepare the input attributes for the NN. (scaler)
# standartization of datasets
sc_X = StandardScaler()
X_trainscaled=sc_X.fit_transform(X_train)
X_testscaled=sc_X.transform(X_test)

sc_y = StandardScaler()
y_trainscaled=sc_y.fit_transform(y_train)
y_testscaled=sc_y.transform(y_test)

# 4. Create the model and optimize the hyper-parameters.
# building the Multi Layer Perceptron
# hyper parameterizing it
model = MLPRegressor(hidden_layer_sizes=(18,3,18),activation="relu", alpha=0.01, random_state=1, max_iter=2000).fit(X_trainscaled, y_trainscaled)

# 5. Evaluate the final model.
# predict and evaluate
y_pred = model.predict(X_testscaled)
print("The Score with (1.0 is awesome)", (r2_score(y_pred, y_testscaled)))

# 6. Save the final model. (optional)
# dump the model
#dump(model, 'model.joblib')
#dump(sc_X, 'sc_X.joblib')
#dump(sc_y, 'sc_y.joblib')