# 19.04.21
# Assignment lab 05

# Master Class: Machine Learning (5MI2018)
# Faculty of Economic Science
# University of Neuchatel (Switzerland)
# Lab 5, see ML21_Exercise_5.pdf for more information

# Authors: 
# - Romain Claret @RomainClaret
# - Sylvain Robert-Nicoud @Nic0uds

1. We choose arbitrarily 1970 and 1990 as years to do the exercise. So we created two dataframes "df_1990" and "df_1970" which contain sub-datasets from the original dataset related to those specific years. Using the pandas_profiling library, we went through the datasets to better understand the attributes' meaning and graphically determine which features are the most meaningful to use to apply clustering techniques on these dataframes. We are aware that using this approach is subjective to our interpretation, but it helped us framing the clusters. Finally, we decided to drop rows with NA values in features from our sub-datasets.

2. Regarding our two dataframes, as the first approach, we used the k-means algorithm. 

3. We had to determine the value for k for clusters; we experimented with k values of 2, 3, 4, and 5. We decided to keep k=4 based on our interpretation and sensitivity of the yield clusters. Note that with k=2 and k=3, observations were very disappeared; with k=5, we noticed two sets with too few elements. Comparing df_1970 and df_1990, using k=4, we can observe that 2 clusters are similar in their location on the plot (bottom-left), with only 1 and 2 outliers belonging to it. However, the clusters in top-left and extreme-right are well-defined.

4. Finally, we decided to try the gaussian-mixture clustering as a second technique because values' distribution within our dataframes seemed to be a good candidate for the algorithm. We see little difference between gaussian-mixture and k-means clustering algorithms but not significantly disparate to be interpreted visually on our selected sub-datasets.