# 10.05.21
# Assignment lab 08

# Master Class: Machine Learning (5MI2018)
# Faculty of Economic Science
# University of Neuchatel (Switzerland)
# Lab 8, see ML21_Exercise_8.pdf for more information

# Authors: 
# - Romain Claret @RomainClaret
# - Sylvain Robert-Nicoud @Nic0uds


# README
- We computed time series on "hc" and "ccon" features because they seemed to carry information in time, explicitly trend information.
- We used a two years window arbitrarily to observe a trend.
- Using the exact hidden layer sizes, our time-series model is less accurate using the r2 score than our model in lab 07. The layers need to be tweaked to provide a better scoring, making sense as the features entering and the expected features extraction are probably different.
- It's probable that our model cannot extract trending information from the window we gave as the dataset is probably too small to build this feature.
- Further steps could be to add manually trending features to help the model find patterns.
