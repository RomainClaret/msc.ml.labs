# 26.04.21
# Assignment lab 06

# Master Class: Machine Learning (5MI2018)
# Faculty of Economic Science
# University of Neuchatel (Switzerland)
# Lab 6, see ML21_Exercise_6.pdf for more information

# Authors: 
# - Romain Claret @RomainClaret
# - Sylvain Robert-Nicoud @Nic0uds

# 1. Use PCA (Principal Component Analysis) to reduce the number of features. Pay special attention to the need of normalization.
We initially used the PCA in assignment 5. In assignment 6, as we are using logarithmic transformation, we are additionally normalizing the data.

# 2. Interpret the results for one year using a decision tree. (with the original fields)

## Composition of our clusters are the following:
- Cluster 0 with 25 items: Australia,Brazil,Canada,China,Colombia,Germany,Spain,France,United Kingdom,Indonesia,India,Iran (Islamic Republic of),Italy,Japan,Republic of Korea,Mexico,Netherlands,Philippines,Poland,Saudi Arabia,Thailand,Turkey,Taiwan,United States,South Africa
- Cluster 1 with 25 items : Bahrain,Bolivia (Plurinational State of),Barbados,Botswana,Cyprus,Fiji,Honduras,Jamaica,Jordan,Lesotho,Luxembourg,"China, Macao SAR",Malta,Mongolia,Mozambique,Mauritania,Mauritius,Namibia,Nicaragua,Panama,Qatar,Eswatini,Trinidad and Tobago,Uruguay,Zambia
- Cluster 2  with 35 items: Angola,Argentina,Austria,Belgium,Bulgaria,Switzerland,Denmark,Dominican Republic,Ecuador,Egypt,Finland,Greece,Guatemala,"China, Hong Kong SAR",Hungary,Ireland,Iraq,Israel,Kenya,Kuwait,Sri Lanka,Morocco,Malaysia,Nigeria,Norway,New Zealand,Peru,Portugal,Romania,Sudan,Singapore,Sweden,Tunisia,Venezuela (Bolivarian Republic of),Zimbabwe
- Cluster 3  with 18 items: Burundi,Benin,Burkina Faso,Central African Republic,Chile,Côte d'Ivoire,Cameroon,Costa Rica,Gabon,Iceland,Lao People's DR,Niger,Paraguay,Rwanda,Senegal,Sierra Leone,Togo,U.R. of Tanzania: Mainland

Surprisingly to our understanding, we noticed that France, Germany, and Iran are in the same cluster. And that Switzerland is in the same cluster as Angola, Morocco, Nigeria, and Hong Kong.
It appears that the clusters' composition is not clear to our understanding at the first view. Applying a decision tree gave us the following features that are used as branches to classify the data.

- In cluster 0, we have 2 conditions :
    ccon (Real consumption of households and government) > 12.165
    ccon <= 12.165 & xr (Exchange rate, national currency/USD) > 3.899 & rgdpe (Expenditure-side real GDP at chained PPPs) > 12.007

- In cluster 1, we have 1 condition :
    ccon <= 12.165 & xr <= 3.899 & rgdpe <= 10.392

- In cluster 2, we have 2 conditions :
    ccon <= 12.165 & xr <= 3.899 & rgdpe > 10.392
    ccon <= 12.165 & xr  > 3.899 & rgdpe <= 12.007 & rconna (Real consumption at constant 2017 national prices )> 11.57

- In cluster 3, we have 1 condition :
    ccon <= 12.165 & xr  > 3.899 & rgdpe <= 12.007 & rconna <= 11.57

# 3. Compare the results to the clusters from exercise 5.
We noticed that the clusters' compositions are not the same.

- One reason to explain this differences, is that we did not use the same features to construct the clusters. Indeed, in assigment 5 we only used visually well spread features: "country","hc","ctfp","cwtfp","delta","pl_con","pl_da","pl_gdpo","csh_g","pl_c","pl_i","pl_g") for the classification. In assignment 6, we used additional features: "rgdpe","pop","ccon","rgdpna","rconna","xr" and applied a logarythmic transformation on those features.
Based on the result that the items in the clusters are not the same, we assume that ignored features in assignment 5 are significant for the classification, even if the item's distribution in clusters seems weird to us in both assignments.

We also tried to exclude outlier values based on thresholds from the PCA representation, but the results didn't seem to impact the clusters, so we decided not to exclude any values.