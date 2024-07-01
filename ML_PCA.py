import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  
scaler = StandardScaler()
import matplotlib.pyplot as plt
import scipy

feature_path="./dataset/PCA/"
Cor_descriptor_data = "Cor_descriptor.pkl"
data = pd.read_pickle(feature_path + Cor_descriptor_data)
data = data.drop(columns=['ID'])  
  
X = data.drop(columns=['viscosity'])
y = data['viscosity'] 
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

for i in [X_train, X_test, y_train,y_test]:
    i.index = range(i.shape[0])
scaler.fit(X_train)
#scaler.fit(X_test)
X = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_cov = (X.T @ X) / (X.shape[0] - 1) #here we are taking our mean as 0 

P_values, P =scipy.linalg.eig(X_cov)
P_values=P_values.real
P=P.real

idx = np.argsort(P_values, axis=0)[::-1]
cumsum = np.cumsum(P_values[idx]) / np.sum(P_values[idx])
print("===PCA plot data===")
print(cumsum)
print("===========")

k = None  
for i in range(1, len(cumsum)):  
    if cumsum[i] > 0.95:  
        k = i  
        break

k = np.argmax(cumsum >= 0.95) + 1 

if k is None:  
    raise ValueError("==There is not enough principal component accumulation to explain 95% of the variance==")  
  
print(f"==Dimensionality is reduced using the first {k} principal components==")  
  
X_pca = X.dot(P[:, :k]) 
print(X_pca) 
X_test_pca = X_test.dot(P[:, :k])  
print(X_test_pca)
print(X_test_pca.shape)

X_train_with_viscosity = pd.concat([y_train.to_frame('viscosity'), pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(k)])], axis=1)  
X_test_with_viscosity = pd.concat([y_test.to_frame('viscosity'), pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(k)])], axis=1)  
  
all_data_with_viscosity = pd.concat([X_train_with_viscosity, X_test_with_viscosity], ignore_index=True)  
  
with open('./dataset/PCA/Opt_PCA_descriptor.pkl', 'wb') as f:  
    pd.to_pickle(all_data_with_viscosity, f)

print ("***==========Plots of principal components VS cumulative variance==========***")  
plt.figure(figsize=(10,8))
xint = range(1, len(cumsum) + 1)
plt.plot(xint, cumsum)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.xticks(xint)
plt.savefig('./figure/PCA/Cumulative_explained_variance.png')

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


#Definition of RF model for hyperparameters optimization
def RF(n_estimator,max_depths,min_samples_split,min_samples_leaf):
    n_estimator = int(n_estimator)
    max_depths = int(max_depths)
    min_samples_split=int(np.round(min_samples_split))
    min_samples_leaf=int(np.round(min_samples_leaf))
    rfg = RandomForestRegressor(n_estimators = n_estimator, max_features='auto',random_state=1, max_depth = max_depths,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
    rfg.fit(X_pca,y_train.values.ravel())
    res = rfg.predict(X_test_pca)
    print("Training set score: %f" % rfg.score(X_pca,y_train))
    print("Test set score: %f" % rfg.score(X_test_pca,y_test))
    error1=rfg.score(X_pca,y_train)
    error2=rfg.score(X_test_pca,y_test)
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
    model.fit(X_pca,y_train.values.ravel())
    res = model.predict(X_test_pca)
    print ("kernel_chose:%s" %kernel)
    print("Training set score: %f" % model.score(X_pca,y_train))
    print("Test set score: %f" % model.score(X_test_pca,y_test))
    error1=model.score(X_pca,y_train)
    error2=model.score(X_test_pca,y_test)
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
    clf.fit(X_pca,y_train.values.ravel())
    res = clf.predict(X_test_pca)
    print("solver: %s" % solver)
    print("Training set score: %f" % clf.score(X_pca,y_train))
    print("Test set score: %f" % clf.score(X_test_pca,y_test))
    error=clf.score(X_test_pca,y_test)
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
    XGboost.fit(X_pca,y_train.values.ravel())
    res = XGboost.predict(X_test_pca)

    print("Training set score: %f" % XGboost.score(X_pca,y_train))
    print("Test set score: %f" % XGboost.score(X_test_pca,y_test))
    error1=XGboost.score(X_pca,y_train)
    error2=XGboost.score(X_test_pca,y_test)
    error=error2
    return error


#Note: Since Bayesian optimization is affected by the initial random candidates, it may cause the optimization hyperparameters to vary slightly from different machines
from bayes_opt import BayesianOptimization
pbounds = {"learning_rate": (0.01, 0.3),"n_estimators": (100, 1500),"max_depth":(3,10),"min_child_weight":(1,10),"subsample":(0.1,1),"colsample_bytree": (0.3,1),"gamma":(0,10),"reg_alpha":(0,10),"reg_lambda":(1,10)}
optimizer = BayesianOptimization(f=XGboost,pbounds=pbounds,random_state=4)
bo=optimizer.maximize(init_points=10,n_iter=400) # n_iter=400
print(optimizer.max)

#The set of hyperparameters of the XGboost model
XGboost_par=optimizer.max.get('params')
XGboost_par

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
XGboost.fit(X_pca,y_train.values.ravel())
res = XGboost.predict(X_test_pca)
print("Training set score: %f" % XGboost.score(X_pca,y_train))
print("Test set score: %f" % XGboost.score(X_test_pca,y_test))
mse=mean_squared_error(y_test,res)
rmse=math.sqrt(mse)
print('MSE:{}'.format(mse))
print('RMSE:{}'.format(rmse))
XGboost_train= XGboost.predict(X_pca)
XGboost_test= XGboost.predict(X_test_pca)


#Note: Since Bayesian optimization is affected by the initial random candidates, it may cause the optimization hyperparameters to vary slightly from different machines
from bayes_opt import BayesianOptimization
pbounds = {"n_estimator": (10, 2000),"max_depths":(10,25),"min_samples_split":(2,10),"min_samples_leaf":(1,10)}
optimizer = BayesianOptimization(f=RF,pbounds=pbounds,random_state=2)
bo=optimizer.maximize(init_points=10,n_iter=400) # n_iter=400
print(optimizer.max)

#The set of hyperparameters of the RF model
RF_par=optimizer.max.get('params')
print(RF_par)

#RF model
print ("***==========Construction of RF model==========***")
n_estimator = int(RF_par.get('n_estimator'))
max_depths = int(RF_par.get('max_depths'))
min_samples_split=int(np.round(RF_par.get('min_samples_split')))
min_samples_leaf=int(np.round(RF_par.get('min_samples_leaf')))
rfg = RandomForestRegressor(n_estimators = n_estimator, max_features='auto',random_state=1, max_depth = max_depths,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
rfg.fit(X_pca,y_train.values.ravel())
res = rfg.predict(X_test_pca)
print("Training set score: %f" % rfg.score(X_pca,y_train))
print("Test set score: %f" % rfg.score(X_test_pca,y_test))
mse=mean_squared_error(y_test,res)
rmse=math.sqrt(mse)
print('MSE:{}'.format(mse))
print('RMSE:{}'.format(rmse))
RF_train= rfg.predict(X_pca)
RF_test= rfg.predict(X_test_pca)


#Note: Since Bayesian optimization is affected by the initial random candidates, it may cause the optimization hyperparameters to vary slightly from different machines
pbounds = {'kernel_chose':(0,4), 'alpha':(1e-3,1), 'gamma':(0.0001,0.01) ,'degree':(2,5),'coef0':(1,5) }
optimizer_krr= BayesianOptimization(f=KRR,pbounds=pbounds,random_state=2)
bo=optimizer_krr.maximize(init_points=10,n_iter=400) # n_iter=400
print(optimizer_krr.max)

#The set of hyperparameters of the KRR model
KRR_par=optimizer_krr.max.get('params')
print(KRR_par)

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
model.fit(X_pca,y_train.values.ravel())
res = model.predict(X_test_pca)

KRR_train_score = model.score(X_pca,y_train)  
KRR_test_score = model.score(X_test_pca,y_test)

print("Training set score: %f" % model.score(X_pca,y_train))
print("Test set score: %f" % model.score(X_test_pca,y_test))
KRR_train = model.predict(X_pca)
KRR_test = model.predict(X_test_pca)
KRR_mse=mean_squared_error(y_test,KRR_test)
KRR_rmse=math.sqrt(mse)
print("MSE: %f" % KRR_mse)
print("RMSE: %f" % KRR_rmse)


#Note: Since Bayesian optimization is affected by the initial random candidates, it may cause the optimization hyperparameters to vary slightly from different machines
pbounds = {'alphas': (1e-5, 1),"node":(1,256),"number_hidden_layer":(1,5),'solver_chose':(0,3)}
optimizer_MLP = BayesianOptimization(f=MLP,pbounds=pbounds,random_state=1)
bo=optimizer_MLP.maximize(init_points=10,n_iter=400) # n_iter=400
print(optimizer_MLP.max)

#The set of hyperparameters of the MLP model
MLP_par=optimizer_MLP.max.get('params')

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
layer_sizes=[nodes,nodes,nodes,nodes][0:number_layer:1]
print(layer_sizes)
print(alphas)
clf = MLPRegressor(solver=solver,alpha=alphas,
                   hidden_layer_sizes=layer_sizes, max_iter=5000,
                   verbose=False, tol=0.0001,random_state=randomseed)
clf.fit(X_pca,y_train.values.ravel())
print("Training set score: %f" % clf.score(X_pca,y_train))
print("Test set score: %f" % clf.score(X_test_pca,y_test))
MLP_train = clf.predict(X_pca)
MLP_test = clf.predict(X_test_pca)
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
plt.savefig('./figure/PCA/RF.png')
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
#plt.title("Traing dataset with r2=",font1)

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

plt.savefig('./figure/PCA/KRR.png')
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
plt.savefig('./figure/PCA/MLP.png')
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
#plt.title("Traing dataset with r2=",font1)

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
plt.savefig('./figure/PCA/XGboost.png')
#plt.show()


path1="./pretrained_model/PCA/"
RF_name = 'RF_Opt.model'
KRR_name = 'KRR_Opt.model'
MLP_name = 'MLP_Opt.model'
XGboost_name = 'XGboost_Opt.model'
joblib.dump(rfg, path1+RF_name)
joblib.dump(model, path1+KRR_name)
joblib.dump(clf, path1+MLP_name)
joblib.dump(XGboost, path1+XGboost_name)

