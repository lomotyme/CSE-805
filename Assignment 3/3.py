Appendix
#Import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#Load Dataset
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
my_csv = "pima-indians-diabetes_Gokaraju_edited.csv"
pima_edited = pd.read_csv(my_csv, names = names)

#Question 3
pima_edited.describe()

#Question 4
indice = np.where(pima_edited == 99999)
print(list(zip(indice[0], indice[1])))
#indice = np.asarray(pima_edited == 99999).nonzero()
#print(list(zip(indice[0], indice[1])))

#Question 5
pima_missingData=pima_edited.drop(columns = ['preg', 'class'])
print("Missing values as zero:\n",(pima_missingData==0).sum(axis = 0), "\n")
print("Missing values as 99999:\n",(pima_missingData == 99999).sum(axis = 0))

#Question 6
for (colname, coldata) in pima_edited.drop(columns = 'class').iteritems():
    mean, stdev, threeSigma= np.mean(coldata.values), np.std(coldata.values), 3*stdev
    lower, upper = mean - threeSigma, mean + threeSigma
    if (np.nonzero(coldata.values < lower) or np.nonzero(coldata.values > upper)):
        print(colname)
        
#Question 7
pima_edited.drop(columns = 'class').plot(kind = 'box', subplots = True, layout = (3, 3),  fontsize = 20, grid = False)
plt.plot()

#Question 8
pima_replace0 = pima_missingData.replace(0, np.nan)
pima_full_replace= pima_replace0.replace(99999, np.nan)
pima_full_replace.describe()

#Question 9
pima_noNan = pima_edited[~np.isnan(pima_full_replace).any(axis = 1)]
pima_noNan.plot(kind = 'box', subplots = True, layout = (3, 3),  fontsize = 20, grid = False)
plt.plot()

#Question 10
array = pima_edited.values
X, Y = array[:, 0:8], array[:, 8]
model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)


#Question 11
#array = pima_edited.values
array_noNan = pima_noNan.values
#X, Y = array[:, 0:8], array[:, 8]
X_noNan, Y_noNan = array_noNan[:, 0:8], array_noNan[:, 8]
kfold = KFold(n_splits = 10, shuffle = True, random_state = 7)
model = SVC()
results = cross_val_score(model, X, Y, cv=kfold)
results_noNan = cross_val_score(model, X_noNan, Y_noNan, cv=kfold)
print(results.mean(), results_noNan.mean())

#Question 12
pima_att_removed = pima_edited.drop(columns = ['test', 'skin'])
array_att_removed = pima_att_removed.values
X_att_removed, Y_att_removed = array_att_removed[:, 0:6], array_att_removed[:, 6]
results_att_removed = cross_val_score(model, X_att_removed, Y_att_removed, cv=kfold)
print(results.mean(), results_att_removed.mean())

#Question 13
pima_edited1= pima_edited.replace(99999, 0)
pima_mean = pima_edited1
pima_mean['plas'].replace(0, pima_edited1['plas'].mean(), inplace = True)
pima_mean['pres'].replace(0, pima_edited1['pres'].mean(), inplace = True)
pima_mean['skin'].replace(0, pima_edited1['skin'].mean(), inplace = True)
pima_mean['test'].replace(0, pima_edited1['test'].mean(), inplace = True)
pima_mean['mass'].replace(0, pima_edited1['mass'].mean(), inplace = True)

array_mean = pima_edited4.values
X_mean, Y_mean = array_mean[:, 0:8], array_mean[:, 8]
results_mean = cross_val_score(model, X_mean, Y_mean, cv=kfold)
print(results.mean(), results_mean.mean())

#Question 14
scaler = MinMaxScaler()
pima_edited4 = pima_mean.drop(columns = ('pedi'))
scaler.fit(pima_edited4)
edited_4 = scaler.transform(pima_edited4)
pima_normalized = pd.DataFrame(edited_4, columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'age', 'class'])
pima_normalized.insert(6, 'pedi', np.asarray(pima_mean.pedi))

array_norm = pima_normalized.values
X_norm, Y_norm = array_norm[:, 0:8], array_norm[:, 8]
results_norm = cross_val_score(model, X_norm, Y_norm, cv=kfold)
print(results.mean(), results_norm.mean())

#Question 15
scaler2 = StandardScaler()
scalr = scaler2.fit(pima_mean[['plas', 'pres']])
scaled = scalr.transform(pima_mean[['plas', 'pres']])
pima_scaled = pd.DataFrame(scaled, columns = ['plas', 'pres'])

drop_plas = pima_mean.drop(columns = 'plas')
pima_standard = drop_plas.drop(columns = 'pres')
pima_standard.insert(1, 'plas', np.asarray(pima_scaled.plas))
pima_standard.insert(2, 'pres', np.asarray(pima_scaled.pres))

array_std = pima_standard.values
X_std, Y_std = array_std[:, 0:8], array_std[:, 8]
results_std = cross_val_score(model, X_std, Y_std, cv=kfold)
print(results.mean(), results_std.mean())