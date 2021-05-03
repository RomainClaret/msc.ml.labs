# 26.04.21
# Assignment lab 06

# Master Class: Machine Learning (5MI2018)
# Faculty of Economic Science
# University of Neuchatel (Switzerland)
# Lab 6, see ML21_Exercise_6.pdf for more information

# Authors: 
# - Romain Claret @RomainClaret
# - Sylvain Robert-Nicoud @Nic0uds
1. Use PCA (Principal Component Analysis) to reduce the number of features. Pay special
attention to the need of normalization.
Already done in assignment 5

2. Interpret the results for one year using a decision tree. (with the original fields)
Composition of our clusters are the following:
    0: Australia,Brazil,Canada,China,Colombia,Germany,Spain,France,United Kingdom,Indonesia,India,Iran (Islamic Republic of),Italy,Japan,Republic of Korea,Mexico,Netherlands,Philippines,Poland,Saudi Arabia,Thailand,Turkey,Taiwan,United States,South Africa,,,,,,,,,,
    1: Bahrain,Bolivia (Plurinational State of),Barbados,Botswana,Cyprus,Fiji,Honduras,Jamaica,Jordan,Lesotho,Luxembourg,"China, Macao SAR",Malta,Mongolia,Mozambique,Mauritania,Mauritius,Namibia,Nicaragua,Panama,Qatar,Eswatini,Trinidad and Tobago,Uruguay,Zambia,,,,,,,,,,
    2: Angola,Argentina,Austria,Belgium,Bulgaria,Switzerland,Denmark,Dominican Republic,Ecuador,Egypt,Finland,Greece,Guatemala,"China, Hong Kong SAR",Hungary,Ireland,Iraq,Israel,Kenya,Kuwait,Sri Lanka,Morocco,Malaysia,Nigeria,Norway,New Zealand,Peru,Portugal,Romania,Sudan,Singapore,Sweden,Tunisia,Venezuela (Bolivarian Republic of),Zimbabwe
    3: Burundi,Benin,Burkina Faso,Central African Republic,Chile,Côte d'Ivoire,Cameroon,Costa Rica,Gabon,Iceland,Lao People's DR,Niger,Paraguay,Rwanda,Senegal,Sierra Leone,Togo,U.R. of Tanzania: Mainland,,,,,,,,,,,,,,,,,
We notice that France, Germany and Iran are in the same cluster. Switzerland is in the same cluster as Angola, Morocco, Nigeria and Hong Kong.
Composition of cluster is not clear at the first view so here are the features that made the clusters
In cluster 0 we have 2 conditions :
    ccon (Real consumption of households and government)> 12.165
    ccon <= 12.165 & xr (Exchange rate, national currency/USD) > 3.899 & rgdpe (Expenditure-side real GDP at chained PPPs) > 12.007
In cluster 1 we have 1 condition :
    ccon <= 12.165 & xr <= 3.899 & rgdpe <= 10.392
In cluster 2 we have 2 conditions :
    ccon <= 12.165 & xr <= 3.899 & rgdpe > 10.392
    ccon <= 12.165 & xr  > 3.899 & rgdpe <= 12.007 & rconna (Real consumption at constant 2017 national prices )> 11.57
In cluster 3 we have 1 condition :
    ccon <= 12.165 & xr  > 3.899 & rgdpe <= 12.007 & rconna <= 11.57

3. Compare the results to the clusters from exercise 5.
Cluster's composition is not the same 
A reason that can explain this defferences is that we did not use the same features to construct our clusters.
(in assigment 5 we used "country","hc","ctfp","cwtfp","delta","pl_con","pl_da","pl_gdpo","csh_g","pl_c","pl_i","pl_g") using only the PCA on these. 
in assignment 6 we used additional features : "rgdpe","pop","ccon","rgdpna","rconna","xr" on these we performed a logarythmic transformation)
we excluded these because these had some outlier's values and a singular distribution so we decided not to use it.
But as we can see in our DTree, differences in the additional features seems to be more significant than differences in the features used in assignment 5)

It's important to see well look at the data even if distribution seems weird. We could have pass away important informations.