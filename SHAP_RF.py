import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.model_selection import train_test_split

feature_path="./dataset/"
Cor_descriptor_data = "Opt_RFE_descriptor.pkl"
data = pd.read_pickle(feature_path + Cor_descriptor_data)
data = data.drop(columns=['ID'])  
  
X = data.drop(columns=['viscosity'])
y = data['viscosity'] 
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

#select_feature=['T', 'Rg', 'K_bond_ave', 'K_ang_ave', 'Theta0_ave', 'AATS1dv',
#       'AATSC3dv', 'HybRatio', 'ETA_dBeta', 'IC1', 'nHRing', 'TSRW10']
select_feature=['K_bond_ave', 'K_ang_ave', 'Theta0_ave', 'AATS1dv',
       'AATSC3dv', 'HybRatio', 'ETA_dBeta', 'IC1', 'nHRing', 'TSRW10']

X_train=X_train[select_feature]
X_test=X_test[select_feature]

for i in [X_train, X_test, y_train,y_test]:
    i.index = range(i.shape[0])

scaler.fit(X_train)
X = scaler.transform(X_train)
X_test = scaler.transform(X_test)

feature_name_list =X_train.columns.values.tolist() 
X_FEATURE = pd.DataFrame(X,columns=feature_name_list)
print(X_FEATURE) 


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math
import numpy as np

#RF model
print ("***==========Construction of RF model==========***")
n_estimator = int(1049.5194149022827)
max_depths = int(19.120521400013587)
min_samples_split=int(np.round(4.187617655312546))
min_samples_leaf=int(np.round(2.2678063266759345))
print(n_estimator,max_depths,min_samples_split,min_samples_leaf)
rfg = RandomForestRegressor(n_estimators = n_estimator, max_features='auto',random_state=1, max_depth = max_depths,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
rfg.fit(X,y_train.values.ravel())
res = rfg.predict(X_test)
print("Training set score: %f" % rfg.score(X,y_train))
print("Test set score: %f" % rfg.score(X_test,y_test))
mse=mean_squared_error(y_test,res)
rmse=math.sqrt(mse)
print('MSE:{}'.format(mse))
print('RMSE:{}'.format(rmse))

import shap
import matplotlib.pyplot as plt
shap.initjs()

explainer = shap.TreeExplainer(rfg)

shap_values = explainer.shap_values(X)
ax =shap.summary_plot(shap_values, X_FEATURE , plot_type="bar", title='Train', max_display=12, show=False, color="k")
ax = plt.gca()
labels = [l.get_text() for l in ax.get_yticklabels()]
values = [rect.get_width() for rect in ax.patches]
df_feature = pd.DataFrame({'Labels': labels, 'Values': values})

print("***==========Relative importance values==========***")
print(df_feature)


from matplotlib.lines import Line2D
importances = rfg.feature_importances_
print("***==========Number of features==========***")
print(len(importances))

indices = np.argsort(importances)
features =X_train.columns
plt.figure(figsize=(8,5))
plt.rc('font',family='Times New Roman',weight='normal')
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}
#pyhiscalfeature=['Rg', 'K_bond_ave', 'K_ang_ave', 'Theta0_ave']
pyhiscalfeature=['T', 'Rg', 'K_bond_ave', 'K_ang_ave', 'Theta0_ave']
#plt.title('COE Feature importances')
colorbar=['#788402','#2F365F']
colorlist1=[]
for feature in df_feature["Labels"]:
    if feature in pyhiscalfeature:
        colorlist1.append(colorbar[0])
    else:
        colorlist1.append(colorbar[1]) 

plt.bar(range(len(df_feature)),df_feature["Values"][::-1],color=colorlist1[::-1], align='center')  
plt.xticks(range(len(indices)),df_feature["Labels"][::-1], rotation=60,fontsize=16)
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6], fontsize=14)
plt.ylabel('Relative importance values', font1)
#plt.xlabel('descriptor',font1)
descriptor_name=["MD","Mordred"]
label=[1,2]
legend_elements = []
legend_font = {'family' : 'Times New Roman', 'weight' : 'normal', 'size': 16}
for label, name in enumerate(descriptor_name):
    legend_elements.append(Line2D(
        [0], [0], marker='s', lw=0, markersize=15,
        color=colorbar[label], label=name, alpha=0.8))
    plt.legend(prop=legend_font,loc="best",borderaxespad=0.2,handles=legend_elements,  frameon=False)

plt.tight_layout()
plt.savefig('./figure/shap/shap_importance.png')
#plt.show()
plt.close()


import matplotlib.cm as cm
plt.figure(figsize=(6,9.5))
ax=shap.summary_plot(shap_values,X_FEATURE,max_display = 50, show=False, cmap=cm.coolwarm, plot_size=(5,5),sort=False)
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}
#cb=plt.colorbar(ax)
#cb.ax.tick_params(labelsize=14,length=10)
plt.rc('font', family='Times New Roman')
plt.rcParams.update({'font.size':16})

plt.tight_layout()
plt.xticks(fontproperties ='Times New Roman', fontsize=16)
plt.yticks(fontproperties ='Times New Roman', fontsize=16,rotation=30)
plt.xlabel('SHAP value',font1)
plt.tight_layout()
plt.savefig('./figure/shap/shap_value.png')
#plt.show()
plt.close()


plt.figure(figsize=(20,9.5))
shap_interaction_values = explainer.shap_interaction_values(X_FEATURE)
shap.summary_plot(shap_interaction_values,X_FEATURE)
plt.savefig('./figure/shap/shap_interaction_values.png')
plt.close()


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'Times New Roman' 
shap_interaction_values = explainer.shap_interaction_values(X_FEATURE)
for fea in feature_name_list:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    shap.dependence_plot(
        (fea, fea),
        shap_interaction_values, X_train,
        display_features=X_train,
        cmap='coolwarm', show=False, ax=ax
    )

    xdata = np.array(ax.collections[0].get_offsets())
    ydata = np.ravel(xdata[:, 1])
    xdata = np.ravel(xdata[:, 0])

    plt.close(fig) 

    plt.figure(figsize=(7,6))
    plt.scatter(x=xdata, y=ydata, color='#000080', marker='o', s=70, alpha=0.8, edgecolor='#C0C0C0', linewidths=1)

    for spine in ax.spines.values():  
        spine.set_linewidth(3)  

    ax.xaxis.set_tick_params(labelsize=16)  
    ax.yaxis.set_tick_params(labelsize=16) 

    plt.xlabel(fea, fontsize=24)
    plt.ylabel('SHAP value', fontsize=24)

    plt.tick_params(axis='x', labelsize=16)  
    plt.tick_params(axis='y', labelsize=16)

    plt.tight_layout()

    plt.savefig('./figure/shap/' 'shap_' + fea + '.png')
    plt.close()


