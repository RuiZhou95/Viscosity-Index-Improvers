#import ipdb

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
from sklearn.model_selection import train_test_split
import pickle  
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import math
from bayes_opt import BayesianOptimization
scaler = StandardScaler()

path="./dataset/"
file='Cor_descriptor.pkl'
with open(path+file, 'rb') as f:  
    dataset = pickle.load(f) 
  
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)  
  
with open(path + 'Cor_train.pkl', 'wb') as train_file:  
    pickle.dump(train_data, train_file) 
  
with open(path + 'Cor_test.pkl', 'wb') as test_file:  
    pickle.dump(test_data, test_file)

file='Cor_train.pkl'
train_df= pd.read_pickle(path+file)
print(train_df)

file1='Cor_test.pkl'
test_df= pd.read_pickle(path+file1)
print(test_df)

#ipdb.set_trace() # ipdb debug break point

X_train=train_df.drop(train_df.columns[0:2], axis=1)
y_train=train_df["viscosity"]
X_test=test_df.drop(test_df.columns[0:2], axis=1)
y_test=test_df["viscosity"]

select_feature=['T', 'K_bond_ave', 'K_ang_ave', 'Theta0_ave', 'AATS1dv', 'AATSC3dv', 'HybRatio', 'ETA_dBeta', 'IC1', 'nHRing', 'TSRW10']

X_selecttrain=X_train[select_feature]
X_selecttest=X_test[select_feature]
for i in [X_selecttrain,X_selecttest, y_train, y_test]:
    i.index = range(i.shape[0])
scaler.fit(X_selecttrain)
Xtrain  = scaler.transform(X_selecttrain)
Xtest = scaler.transform(X_selecttest)

"""
#Definition of RF model for hyperparameters optimization
def RF(n_estimator,max_depths,min_samples_split,min_samples_leaf):
    n_estimator = int(n_estimator)
    max_depths = int(max_depths)
    min_samples_split=int(np.round(min_samples_split))
    min_samples_leaf=int(np.round(min_samples_leaf))
    rfg = RandomForestRegressor(n_estimators = n_estimator, max_features='auto',random_state=1, max_depth = max_depths,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
    rfg.fit(Xtrain,y_train.values.ravel())
    res = rfg.predict(Xtest)
    print("Training set score: %f" % rfg.score(Xtrain,y_train))
    print("Test set score: %f" % rfg.score(Xtest,y_test))
    error1=rfg.score(Xtrain,y_train)
    error2=rfg.score(Xtest,y_test)
    error=error2
    return error


#Definition of KRR model for hyperparameters optimization
def KRR(kernel_chose,alpha,gamma,degree,coef0):
    if kernel_chose<=1:
        kernel="rbf"
    elif kernel_chose>1 and kernel_chose<=2:
        kernel='laplacian'
    elif kernel_chose>2 and kernel_chose<=3:
        kernel='polynomial'
    else:
        kernel='sigmoid'
    alpha=round(alpha,5)
    gamma==round(gamma,5)
    degree=round(degree)
    coef0=round(coef0,1)
    model=KernelRidge(kernel=kernel,alpha=alpha,gamma=gamma,degree=degree,coef0=coef0)
    model.fit(Xtrain,y_train.values.ravel())
    res = model.predict(Xtest)
    print ("kernel_chose:%s" %kernel)
    print("Training set score: %f" % model.score(Xtrain,y_train))
    print("Test set score: %f" % model.score(Xtest,y_test))
    error1=model.score(Xtrain,y_train)
    error2=model.score(Xtest,y_test)
    error=error2
    return error


#Definition of MLP model for hyperparameters optimization
def MLP(alphas,node,number_hidden_layer,solver_chose):
    randomseed = np.random.seed(0)
    if solver_chose<=1:
        solver='adam'
    elif solver_chose>1 and solver_chose<=2:
        solver='lbfgs'
    else:
        solver='sgd'
    alphas=alphas
    nodes=int(node)
    number_layer=int(number_hidden_layer)
    layer_sizes=[nodes,nodes,nodes,nodes,nodes,nodes,nodes,nodes,nodes,nodes,nodes,nodes][0:number_layer:1]
    print(layer_sizes)
    print(alphas)
    clf = MLPRegressor(solver=solver,alpha=alphas,
                       hidden_layer_sizes=layer_sizes, max_iter=5000,
                       verbose=False, tol=0.0001,random_state=randomseed)
    clf.fit(Xtrain,y_train.values.ravel())
    res = clf.predict(Xtest)
    print("solver: %s" % solver)
    print("Training set score: %f" % clf.score(Xtrain,y_train))
    print("Test set score: %f" % clf.score(Xtest,y_test))
    error=clf.score(Xtest,y_test)
    return error


#Definition of XGboost model for hyperparameters optimization
def XGboost(learning_rate,n_estimators,max_depth,min_child_weight,subsample,colsample_bytree,gamma,reg_alpha,reg_lambda):
    learning_rate = round( learning_rate, 5)
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_child_weight = int(min_child_weight)
    subsample = round(subsample, 3)
    colsample_bytree = round(colsample_bytree, 3)
    gamma = round(gamma, 3)
    reg_alpha = round(reg_alpha, 3)
    reg_lambda = round(reg_lambda, 3)
    XGboost = xgb.XGBRegressor(learning_rate=learning_rate,n_estimators=n_estimators,max_depth=max_depth,min_child_weight=min_child_weight\
                         ,seed=7,subsample=subsample,colsample_bytree=colsample_bytree,gamma=gamma,reg_alpha=reg_alpha,reg_lambda=reg_lambda,random_state=4)
    XGboost.fit(Xtrain,y_train.values.ravel())
    res = XGboost.predict(Xtest)

    print("Training set score: %f" % XGboost.score(Xtrain,y_train))
    print("Test set score: %f" % XGboost.score(Xtest,y_test))
    error1=XGboost.score(Xtrain,y_train)
    error2=XGboost.score(Xtest,y_test)
    error=error2
    return error


#Note: Since Bayesian optimization is affected by the initial random candidates, it may cause the optimization hyperparameters to vary slightly from different machines
from bayes_opt import BayesianOptimization
pbounds = {"learning_rate": (0.01, 0.3),"n_estimators": (100, 1500),"max_depth":(3,10),"min_child_weight":(1,10),"subsample":(0.1,1),"colsample_bytree": (0.3,1),"gamma":(0,10),"reg_alpha":(0,10),"reg_lambda":(1,10)}
optimizer = BayesianOptimization(f=XGboost,pbounds=pbounds,random_state=4)
bo=optimizer.maximize(init_points=10,n_iter=500) # n_iter=100
print(optimizer.max)

#The set of hyperparameters of the XGboost model
XGboost_par=optimizer.max.get('params')
XGboost_par
"""

#XGboost model
print ("***==========Construction of XGboost model==========***")
learning_rate=round(XGboost_par.get('learning_rate'),3)
n_estimators=int(XGboost_par.get('n_estimators'))
max_depth = int(XGboost_par.get('max_depth'))
min_child_weight=int(XGboost_par.get('min_child_weight'))
subsample=round(XGboost_par.get('subsample'),3)
colsample_bytree=round(XGboost_par.get('colsample_bytree'),3)
gamma=round(XGboost_par.get('gamma'),3)
reg_alpha=round(XGboost_par.get('reg_alpha'),3)
reg_lambda=round(XGboost_par.get('reg_lambda'),3)
print (learning_rate,n_estimators,max_depth,min_child_weight,subsample,colsample_bytree,gamma,reg_alpha,reg_lambda)
XGboost = xgb.XGBRegressor(learning_rate=learning_rate,n_estimators=n_estimators,max_depth=max_depth,min_child_weight=min_child_weight\
                         ,seed=7,subsample=subsample,colsample_bytree=colsample_bytree,gamma=gamma,reg_alpha=reg_alpha,reg_lambda=reg_lambda,random_state=1)
XGboost.fit(Xtrain,y_train.values.ravel())
res = XGboost.predict(Xtest)
print("Training set score: %f" % XGboost.score(Xtrain,y_train))
print("Test set score: %f" % XGboost.score(Xtest,y_test))
mse=mean_squared_error(y_test,res)
rmse=math.sqrt(mse)
print('MSE:{}'.format(mse))
print('RMSE:{}'.format(rmse))
XGboost_train= XGboost.predict(Xtrain)
XGboost_test= XGboost.predict(Xtest)

"""
#Note: Since Bayesian optimization is affected by the initial random candidates, it may cause the optimization hyperparameters to vary slightly from different machines
from bayes_opt import BayesianOptimization
pbounds = {"n_estimator": (10, 2000),"max_depths":(10,25),"min_samples_split":(2,10),"min_samples_leaf":(1,10)}
optimizer = BayesianOptimization(f=RF,pbounds=pbounds,random_state=2)
bo=optimizer.maximize(init_points=10,n_iter=500) # n_iter=100
print(optimizer.max)

#The set of hyperparameters of the RF model
RF_par=optimizer.max.get('params')
RF_par
"""


#RF model
print ("***==========Construction of RF model==========***")
n_estimator = int(RF_par.get('n_estimator'))
max_depths = int(RF_par.get('max_depths'))
min_samples_split=int(np.round(RF_par.get('min_samples_split')))
min_samples_leaf=int(np.round(RF_par.get('min_samples_leaf')))
rfg = RandomForestRegressor(n_estimators = n_estimator, max_features='auto',random_state=1, max_depth = max_depths,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
rfg.fit(Xtrain,y_train.values.ravel())
res = rfg.predict(Xtest)
print("Training set score: %f" % rfg.score(Xtrain,y_train))
print("Test set score: %f" % rfg.score(Xtest,y_test))
mse=mean_squared_error(y_test,res)
rmse=math.sqrt(mse)
print('MSE:{}'.format(mse))
print('RMSE:{}'.format(rmse))
RF_train= rfg.predict(Xtrain)
RF_test= rfg.predict(Xtest)

"""
#Note: Since Bayesian optimization is affected by the initial random candidates, it may cause the optimization hyperparameters to vary slightly from different machines
pbounds = {'kernel_chose':(0,4), 'alpha':(1e-3,1), 'gamma':(0.0001,0.01) ,'degree':(2,5),'coef0':(1,5) }
optimizer_krr= BayesianOptimization(f=KRR,pbounds=pbounds,random_state=2)
bo=optimizer_krr.maximize(init_points=10,n_iter=500) # n_iter=100
print(optimizer_krr.max)


#The set of hyperparameters of the KRR model
KRR_par=optimizer_krr.max.get('params')
print(KRR_par)
"""

#KRR model
print ("***==========Construction of KRR model==========***")
alpha=KRR_par.get('alpha')
coef0=KRR_par.get('coef0')
degree=KRR_par.get('degree')
gamma=KRR_par.get('gamma')
kernel_chose=KRR_par.get('kernel_chose')
if kernel_chose<=1:
    kernel="rbf"
elif kernel_chose>1 and kernel_chose<=2:
    kernel='laplacian'
elif kernel_chose>2 and kernel_chose<=3:
    kernel='polynomial'
else:
    kernel='sigmoid'
alpha=round(alpha,5)
gamma=round(gamma,5)
degree=round(degree)
coef0=round(coef0,1)
model=KernelRidge(kernel=kernel,alpha=alpha,gamma=gamma,degree=degree,coef0=coef0)
model.fit(Xtrain,y_train.values.ravel())

KRR_train_score = model.score(Xtrain,y_train)  
KRR_test_score = model.score(Xtest,y_test)

print("Training set score: %f" % model.score(Xtrain,y_train))
print("Test set score: %f" % model.score(Xtest,y_test))
KRR_train = model.predict(Xtrain)
KRR_test = model.predict(Xtest)
KRR_mse=mean_squared_error(y_test,KRR_test)
KRR_rmse=math.sqrt(mse)
print("MSE: %f" % KRR_mse)
print("RMSE: %f" % KRR_rmse)

"""
#Note: Since Bayesian optimization is affected by the initial random candidates, it may cause the optimization hyperparameters to vary slightly from different machines
pbounds = {'alphas': (1e-5, 1),"node":(1,256),"number_hidden_layer":(1,5),'solver_chose':(0,3)}
optimizer_MLP = BayesianOptimization(f=MLP,pbounds=pbounds,random_state=1)
bo=optimizer_MLP.maximize(init_points=10,n_iter=500)
print(optimizer_MLP.max)

#The set of hyperparameters of the MLP model
MLP_par=optimizer_MLP.max.get('params')
"""

#MLP model (The results differ slightly from our article because different random seed was set up)
print ("***==========Construction of MLP model==========***")
node=MLP_par.get('node')
number_hidden_layer=MLP_par.get('number_hidden_layer')
solver_chose=MLP_par.get('solver_chose')
alphas=MLP_par.get('alphas')
randomseed = np.random.seed(5)
if solver_chose<=1:
    solver='adam'
elif solver_chose>1 and solver_chose<=2:
    solver='lbfgs'
else:
    solver='sgd'
alphas=alphas
nodes=int(node)
number_layer=int(number_hidden_layer)
layer_sizes=[nodes,nodes,nodes,nodes,nodes,nodes,nodes,nodes,nodes,nodes,nodes,nodes][0:number_layer:1]
print(layer_sizes)
print(alphas)
clf = MLPRegressor(solver=solver,alpha=alphas,
                   hidden_layer_sizes=layer_sizes, max_iter=5000,
                   verbose=False, tol=0.0001,random_state=randomseed)
clf.fit(Xtrain,y_train.values.ravel())
print("Training set score: %f" % clf.score(Xtrain,y_train))
print("Test set score: %f" % clf.score(Xtest,y_test))
MLP_train = clf.predict(Xtrain)
MLP_test = clf.predict(Xtest)
mse=mean_squared_error(y_test,MLP_test)
rmse=math.sqrt(mse)
print("MSE: %f" % mse)
print("RMSE: %f" % rmse)

print ("***==========Plots of RF model prediction results==========***")
plt.rc('font',family='Times New Roman',weight='normal')
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}
plt.figure(figsize=(6,5.5))

y_train_flat = y_train.ravel()
RF_train_flat = RF_train.ravel()
y_test_flat = y_test.ravel()
RF_test_flat = RF_test.ravel()

plt.plot(y_train_flat,RF_train_flat,color='#C0C0C0',marker='o',linestyle='', markersize=8, markerfacecolor='#80C149',alpha=1)
plt.plot(y_test_flat,RF_test_flat,color='#C0C0C0',marker='o',linestyle='', markersize=8, markerfacecolor='b',alpha=1)
plt.legend(labels=["Training data","Test data"],loc="lower right",fontsize=18, frameon=False) 
title='Calculated viscosity [mpa.s]'
title1='RF Predicted viscosity [mpa.s]'
plt.xlabel(title,font1)
plt.ylabel(title1,font1)
plt.xlim((5, 44))
plt.ylim((5, 44))
plt.plot([5, 44],[5, 44], color='k', linewidth=1.5, linestyle='--')
my_x_ticks = np.arange(5, 44, 5.0)
my_y_ticks = np.arange(5, 44, 5.0)
plt.xticks(my_x_ticks,size=16)
plt.yticks(my_y_ticks,size=16)
plt.tick_params(width=1.5)
bwith = 1.5 
TK = plt.gca()
TK.spines['bottom'].set_linewidth(bwith)
TK.spines['left'].set_linewidth(bwith)
TK.spines['top'].set_linewidth(bwith)
TK.spines['right'].set_linewidth(bwith)
plt.savefig('./figure/ML/RF.png')
#plt.show()

print ("***==========Plots of KRR model prediction results==========***")
plt.rc('font',family='Times New Roman',weight='normal')
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}
plt.figure(figsize=(6,5.5))
#plt.title("Traing dataset with r2=0.955",font1)

y_train_flat = y_train.ravel()
KRR_train_flat = KRR_train.ravel()
y_test_flat = y_test.ravel()
KRR_test_flat = KRR_test.ravel()

plt.plot(y_train_flat,KRR_train_flat,color='#C0C0C0',marker='o',linestyle='', markersize=8, markerfacecolor='#80C149',alpha=1)
plt.plot(y_test_flat,KRR_test_flat,color='#C0C0C0',marker='o',linestyle='', markersize=8, markerfacecolor='b',alpha=1)
plt.legend(labels=["Training data","Test data"],loc="lower right",fontsize=18, frameon=False) 
title='Calculated viscosity [mpa.s]'
title1='KRR Predicted viscosity [mpa.s]'
plt.xlabel(title,font1)
plt.ylabel(title1,font1)
plt.xlim((5, 44))
plt.ylim((5, 44))
plt.plot([5, 44],[5, 44], color='k', linewidth=1.5, linestyle='--')
my_x_ticks = np.arange(5, 44, 5.0)
my_y_ticks = np.arange(5, 44, 5.0)
plt.xticks(my_x_ticks,size=16)
plt.yticks(my_y_ticks,size=16)
plt.tick_params(width=1.5)
bwith = 1.5
TK = plt.gca()
TK.spines['bottom'].set_linewidth(bwith)
TK.spines['left'].set_linewidth(bwith)
TK.spines['top'].set_linewidth(bwith)
TK.spines['right'].set_linewidth(bwith)

plt.savefig('./figure/ML/KRR.png')
#plt.show()

print ("***==========Plots of MLP model prediction results==========***")
plt.rc('font',family='Times New Roman',weight='normal')
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}
plt.figure(figsize=(6,5.5))
#plt.title("Traing dataset with r2=0.955",font1)

y_train_flat = y_train.ravel()
MLP_train_flat = MLP_train.ravel()
y_test_flat = y_test.ravel()
MLP_test_flat = MLP_test.ravel()

plt.plot(y_train_flat,MLP_train_flat,color='#C0C0C0',marker='o',linestyle='', markersize=8, markerfacecolor='#80C149',alpha=1)
plt.plot(y_test_flat,MLP_test_flat,color='#C0C0C0',marker='o',linestyle='', markersize=8, markerfacecolor='b',alpha=1)
plt.legend(labels=["Training data","Test data"],loc="lower right",fontsize=18, frameon=False) 
title='Calculated viscosity [mpa.s]'
title1='MLP Predicted viscosity [mpa.s]'
plt.xlabel(title,font1)
plt.ylabel(title1,font1)
plt.xlim((5, 44))
plt.ylim((5, 44))
plt.plot([5, 44],[5, 44], color='k', linewidth=1.5, linestyle='--')
my_x_ticks = np.arange(5, 44, 5.0)
my_y_ticks = np.arange(5, 44, 5.0)
plt.xticks(my_x_ticks,size=16)
plt.yticks(my_y_ticks,size=16)
plt.tick_params(width=1.5)
bwith = 1.5
TK = plt.gca()
TK.spines['bottom'].set_linewidth(bwith)
TK.spines['left'].set_linewidth(bwith)
TK.spines['top'].set_linewidth(bwith)
TK.spines['right'].set_linewidth(bwith)
plt.savefig('./figure/ML/MLP.png')
#plt.show()

print ("***==========Plots of XGboost model prediction results==========***")
plt.rc('font',family='Times New Roman',weight='normal')
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}
plt.figure(figsize=(6,5.5))
#plt.title("Traing dataset with r2=0.955",font1)

y_train_flat = y_train.ravel()
XGboost_train_flat = XGboost_train.ravel()
y_test_flat = y_test.ravel()
XGboost_test_flat = XGboost_test.ravel()

plt.plot(y_train_flat,XGboost_train_flat,color='#C0C0C0',marker='o',linestyle='', markersize=8, markerfacecolor='#80C149',alpha=1)
plt.plot(y_test_flat,XGboost_test_flat,color='#C0C0C0',marker='o',linestyle='', markersize=8, markerfacecolor='b',alpha=1)
plt.legend(labels=["Training data","Test data"],loc="lower right",fontsize=18, frameon=False) 
title='Calculated viscosity [mpa.s]'
title1='XGboost Predicted viscosity [mpa.s]'
plt.xlabel(title,font1)
plt.ylabel(title1,font1)
plt.xlim((5, 44))
plt.ylim((5, 44))
plt.plot([5, 44],[5, 44], color='k', linewidth=1.5, linestyle='--')
my_x_ticks = np.arange(5, 44, 5.0)
my_y_ticks = np.arange(5, 44, 5.0)
plt.xticks(my_x_ticks,size=16)
plt.yticks(my_y_ticks,size=16)
plt.tick_params(width=1.5)
bwith = 1.5
TK = plt.gca()
TK.spines['bottom'].set_linewidth(bwith)
TK.spines['left'].set_linewidth(bwith)
TK.spines['top'].set_linewidth(bwith)
TK.spines['right'].set_linewidth(bwith)
plt.savefig('./figure/ML/XGboost.png')
#plt.show()


path1="./pretrained_model/descriptors/"
RF_name = 'RF_Opt.model'
KRR_name = 'KRR_Opt.model'
MLP_name = 'MLP_Opt.model'
XGboost_name = 'XGboost_Opt.model'
joblib.dump(rfg, path1+RF_name)
joblib.dump(model, path1+KRR_name)
joblib.dump(clf, path1+MLP_name)
joblib.dump(XGboost, path1+XGboost_name)