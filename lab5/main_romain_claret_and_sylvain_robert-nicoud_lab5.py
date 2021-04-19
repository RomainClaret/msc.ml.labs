#!/usr/bin/env python3
# 19.04.21
# Assignment lab 05

# Master Class: Machine Learning (5MI2018)
# Faculty of Economic Science
# University of Neuchatel (Switzerland)
# Lab 5, see ML21_Exercise_5.pdf for more information

# https://github.com/RomainClaret/msc.ml.labs

# Authors: 
# - Romain Claret @RomainClaret
# - Sylvain Robert-Nicoud @Nic0uds


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# 1. Import the data and filter the two datasets for the chosen two years.
df = pd.read_stata('pwt100.dta')
df_1990 = df[(df["year"]==1990)]
df_1970 = df[(df["year"]==1970)]

features = [
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
df_1990_cleaned = df_1990[features].dropna()
df_1970_cleaned = df_1970[features].dropna()

# 2. Choose the clustering technique and obtain the clusters for the two datasets.
df_1970_pca = PCA(2).fit_transform(df_1970_cleaned)
df_1990_pca = PCA(2).fit_transform(df_1990_cleaned)
kmeans_1970 = KMeans(n_clusters=4, random_state=42).fit(df_1970_pca)
kmeans_1990 = KMeans(n_clusters=4, random_state=42).fit(df_1990_pca)

# 3. Choose one of the years and visualize the results of the clustering. Give as much detail as possible.
y_kmeans_1970 = kmeans_1970.predict(df_1970_pca)
plt.scatter(df_1970_pca[:,0], df_1970_pca[:,1], c=y_kmeans_1970, s=50, cmap='viridis')
plt.scatter(kmeans_1970.cluster_centers_[:,0], kmeans_1970.cluster_centers_[:,1], c='blue', s=200, alpha=0.9)
plt.show()