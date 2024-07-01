#import ipdb

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

from scipy.spatial.distance import pdist, squareform
from minepy import MINE
from sklearn.preprocessing import StandardScaler
from scipy import stats
import scipy

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

# 1_remove_low_variance_features
file_path= "./dataset/"
file="VII_Descriptors"
df = pd.read_csv(file_path + file + '.csv')
df.to_pickle(file_path + file + '.pkl')
all_data = pd.read_pickle(file_path + file  + '.pkl')
print(all_data)
print("==== Feature engineering stage 0 -> All descriptors  =====")

X = all_data.iloc[:,2:len(all_data)]
y = all_data.iloc[:,0:2]

X_var = pd.DataFrame(X.var())

vt = VarianceThreshold(threshold = 0.1)
X_selected = vt.fit_transform(X)
lowvariance_data = pd.DataFrame(X_selected)

all_name = X.columns.values.tolist()
select_name_index0 = vt.get_support(indices=True)
select_name0 = []
for i in select_name_index0:
    select_name0.append(all_name[i])

lowvariance_data.columns = select_name0
#print(lowvariance_data)
lowvariance_data_y = pd.concat((y,lowvariance_data),axis = 1)
print(lowvariance_data_y)
print("===== Feature engineering stage 1  ======")
print("==== Variance filtering: threshold 0.1  =====")

file1 = r"Var_descriptor.pkl"
lowvariance_data_y.to_pickle(file_path+file1)

all_data=lowvariance_data_y
data = all_data.iloc[:,all_data.columns != "ID"]
descriptor_data = data.iloc[:,data.columns != "viscosity"]
all_data_name_list = list(all_data)
descriptor_name_list = list(descriptor_data)
descriptor_count = len(descriptor_name_list)

#Standardized forms
scaler = StandardScaler()
data_scaler = scaler.fit_transform(data)
DataFrame_data_scaler = pd.DataFrame(data_scaler)

#Set the correlation coefficient parameters
data_pearson = DataFrame_data_scaler.corr(method = 'pearson')
data_spearman = DataFrame_data_scaler.corr(method = 'spearman')
mine = MINE(alpha=0.6, c=15)

#ipdb.set_trace() # ipdb debug break point

# Definition of distance correlation coefficient functions
def distcorr(X, Y):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

column_descriptor = data_scaler.shape[1]

#Pearson correlation
Threshold = 0.05 # 0.1
pearson_correlation_list = []
pearson_pvalue_list = []
pearson_selection_list = []
pearson_list = []
for i in range(1,column_descriptor):
    pearson_correlation_list.append(scipy.stats.pearsonr(data_scaler[:,i],data_scaler[:,0])[0])
    pearson_pvalue_list.append(scipy.stats.pearsonr(data_scaler[:,i],data_scaler[:,0])[1])
    if pearson_pvalue_list[i-1] > Threshold:
        pearson_selection_list.append(0)
    else :
        pearson_selection_list.append(1)
pearson_list.append(pearson_correlation_list)
pearson_list.append(pearson_pvalue_list)
pearson_list.append(pearson_selection_list)

#Spearman correlation
Threshold = 0.05 # 0.1
spearman_correlation_list = []
spearman_pvalue_list = []
spearman_selection_list = []
spearman_list = []
for i in range(1,column_descriptor):
    spearman_correlation_list.append(scipy.stats.spearmanr(data_scaler[:,i],data_scaler[:,0])[0])
    spearman_pvalue_list.append(scipy.stats.spearmanr(data_scaler[:,i],data_scaler[:,0])[1])
    if spearman_pvalue_list[i-1] > Threshold:
        spearman_selection_list.append(0)
    else :
        spearman_selection_list.append(1)
spearman_list.append(spearman_correlation_list)
spearman_list.append(spearman_pvalue_list)
spearman_list.append(spearman_selection_list)

#Distance
Threshold = 0.105 # 0.213
distance_correlation_list = []
distance_selection_list = []
distance_list = []
for i in range(1,column_descriptor):
    distance_correlation_list.append(distcorr(data_scaler[:,i],data_scaler[:,0]))
    if abs(distance_correlation_list[i-1]) <= Threshold:
        distance_selection_list.append(0)
    else :
        distance_selection_list.append(1)
distance_list.append(distance_correlation_list)
distance_list.append(distance_selection_list)

#MIC
Threshold = 0.195 # 0.13
mic_correlation_list = []
mic_selection_list = []
mic_list = []
for i in range(1,column_descriptor):
    mine.compute_score(data_scaler[:,i],data_scaler[:,0])
    mic_correlation_list.append(mine.mic())
    if abs(mic_correlation_list[i-1]) <= Threshold:
        mic_selection_list.append(0)
    else:
        mic_selection_list.append(1)
mic_list.append(mic_correlation_list)
mic_list.append(mic_selection_list)


sum_list = []
for j in range(0,descriptor_count):
    sum_list.append(pearson_selection_list[j] + spearman_selection_list[j] + distance_selection_list[j] + mic_selection_list[j])

sum_selection_list1 = []
Threshold1 = 1
for j in range(0,descriptor_count):
    if sum_list[j] >= Threshold1:
        sum_selection_list1.append(1)
    else:
        sum_selection_list1.append(0)
sum(sum_selection_list1)

sum_selection_list2 = []
Threshold2 = 2
for j in range(0,descriptor_count):
    if sum_list[j] >= Threshold2:
        sum_selection_list2.append(1)
    else:
        sum_selection_list2.append(0)
sum(sum_selection_list2)

sum_selection_list3 = []
Threshold3 = 3
for j in range(0,descriptor_count):
    if sum_list[j] >= Threshold3:
        sum_selection_list3.append(1)
    else:
        sum_selection_list3.append(0)
sum(sum_selection_list3)

sum_selection_list4 = []
Threshold4 = 4
for j in range(0,descriptor_count):
    if sum_list[j] >= Threshold4:
        sum_selection_list4.append(1)
    else:
        sum_selection_list4.append(0)
sum(sum_selection_list4)

#Output
sum_list_all = []
sum_list_all.append(sum_selection_list1)
sum_list_all.append(sum_selection_list2)
sum_list_all.append(sum_selection_list3)
sum_list_all.append(sum_selection_list4)
sum_list_all.append(sum_list)
selection_list_all = []
selection_list_all.append(descriptor_name_list)
selection_list_all.append(pearson_list[0])
selection_list_all.append(pearson_list[1])
selection_list_all.append(pearson_list[2])
selection_list_all.append(spearman_list[0])
selection_list_all.append(spearman_list[1])
selection_list_all.append(spearman_list[2])
selection_list_all.append(distance_list[0])
selection_list_all.append(distance_list[1])
selection_list_all.append(mic_list[0])
selection_list_all.append(mic_list[1])
selection_list_all.append(sum_list_all[0])
selection_list_all.append(sum_list_all[1])
selection_list_all.append(sum_list_all[2])
selection_list_all.append(sum_list_all[3])
selection_list_all.append(sum_list_all[4])
selection_list_all = pd.DataFrame(selection_list_all)


selection_list_all_transpose = selection_list_all.T
selection_list_all_transpose.rename(columns={0:'descriptor_name',1:'pearson_correlation',2:'pearson_pvalue',3:'pearson_selection',
                                             4:'spearman_correlation',5:'spearman_pvalue',6:'spearman_selection',7:'distance_correlation',
                                             8:'distance_selection',9:'mic_correlation',10:'mic_selection',11:'sum_1',
                                             12:'sum_2',13:'sum_3',14:'sum_4',15:'sum'},inplace=True)

filter_data1 = all_data
for k in range(0,len(sum_selection_list1)):
    if sum_selection_list1[k] == 0:
        filter_data1 = filter_data1.drop(descriptor_name_list[k],axis=1)

filter_data2 = all_data
for k in range(0,len(sum_selection_list2)):
    if sum_selection_list2[k] == 0:
        filter_data2 = filter_data2.drop(descriptor_name_list[k],axis=1)

filter_data3 = all_data
for k in range(0,len(sum_selection_list3)):
    if sum_selection_list3[k] == 0:
        filter_data3 = filter_data3.drop(descriptor_name_list[k],axis=1)

filter_data4 = all_data
for k in range(0,len(sum_selection_list4)):
    if sum_selection_list4[k] == 0:
        filter_data4 = filter_data4.drop(descriptor_name_list[k],axis=1)

selection_list_all_transpose.to_csv(file_path+'selection_list_all_transpose.csv')

filter_data1.to_csv(file_path+'filter_threshold_1.csv')
#print(filter_data1)
filter_data2.to_csv(file_path+'filter_threshold_2.csv')
#print(filter_data2)
filter_data3.to_csv(file_path+'filter_threshold_3.csv')
#print(filter_data3)
filter_data4.to_csv(file_path+'Cor_descriptor.csv')
print(filter_data4)
print("===== Feature engineering stage 2  ======")
print("==== Pearson: threshold 0.05   Spearman: threshold 0.05 =====")
print("==== MIC: threshold 0.195       Distance: threshold 0.105  =====")
filter_data4.to_pickle(file_path+'Cor_descriptor.pkl')