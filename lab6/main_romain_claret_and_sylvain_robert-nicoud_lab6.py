#!/usr/bin/env python3
# 26.04.21
# Assignment lab 06

# Master Class: Machine Learning (5MI2018)
# Faculty of Economic Science
# University of Neuchatel (Switzerland)
# Lab 6, see ML21_Exercise_6.pdf for more information

# https://github.com/RomainClaret/msc.ml.labs

# Authors: 
# - Romain Claret @RomainClaret
# - Sylvain Robert-Nicoud @Nic0uds

# 2. Interpret the results for one year using a decision tree. (with the original fields)
# 3. Compare the results to the clusters from exercise 5.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import FunctionTransformer, Normalizer
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree


# Import the data and filter the two datasets for the chosen two years.
df = pd.read_stata('pwt100.dta')
df_1990 = df[(df["year"]==1990)]
df_1970 = df[(df["year"]==1970)]

features = [
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
    "rgdpe",
    "pop",
    "ccon",
    "rgdpna",
    "rconna",
    "xr",
]


# Apply the logarithmic function on badly proportionated features 
log_transformer = FunctionTransformer(np.log1p)

df_1990_log_features = df_1990[features_logarithmic]
df_1990_log = log_transformer.transform(df_1990_log_features)

# Concat logarithmic features with unlogarithmic features
df_1990_concat = pd.concat([df_1990[features], df_1990_log], axis=1, join="inner")

# Drop rows with na values
df_1990_cleaned = df_1990_concat.dropna()

# 1. Pay special attention to the need of normalization.
df_1990_normalized = Normalizer().fit_transform(df_1990_cleaned[features[1:]+features_logarithmic])


# 1 bis. Use PCA (Principal Component Analysis) to reduce the number of features. 
# We choose the kmeans clustering technique and obtain the clusters for the dataset.
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
from sklearn.decomposition import PCA
pca = PCA(2)
df_1990_pca = pca.fit_transform(df_1990_cleaned[features[1:]+features_logarithmic])
kmeans_1990 = KMeans(n_clusters=4, random_state=42).fit(df_1990_pca)

# Visualization of the clustering results.
y_kmeans_1990 = kmeans_1990.predict(df_1990_pca)
plt.scatter(df_1990_pca[:,0], df_1990_pca[:,1], c=y_kmeans_1990, s=50, cmap='viridis')
plt.scatter(kmeans_1990.cluster_centers_[:,0], kmeans_1990.cluster_centers_[:,1], c='blue', s=200, alpha=0.9)
plt.show()

# Listing the countries in the clusters
def country_listing(df, clusters):
    tmp = []
    for i_k in range(0, max(clusters)+1):
        tmp_k = []
        for i_c, c in enumerate(clusters):
            if c == i_k: tmp_k.append(df.iloc[i_c]["country"])
        tmp.append(tmp_k)
    return tmp

df_test = pd.DataFrame.from_records(country_listing(df_1990_cleaned, y_kmeans_1990))
df_test.to_csv("out.csv", index=False)

# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
clf_dtree = DecisionTreeRegressor(random_state=42)
clf_dtree.fit(df_1990_cleaned[features[1:]+features_logarithmic], y_kmeans_1990)

fig = plt.figure(figsize=(10,10))
_ = tree.plot_tree(clf_dtree,
                   feature_names = df_1990_cleaned[features[1:]+features_logarithmic].columns,
                   class_names=y_kmeans_1990,
                   filled=True)

fig.savefig("decistion_tree.png")

for e in country_listing(df_1990_cleaned, y_kmeans_1990):
    print(len(e))

#ccon, Real consumption of households and government, at current PPPs (in mil. 2017US$)
#rgdpe, Expenditure-side real GDP at chained PPPs (in mil. 2017US$)
#rconna, Real consumption at constant 2017 national prices (in mil. 2017US$)
#xr, Exchange rate, national currency/USD (market+estimated)