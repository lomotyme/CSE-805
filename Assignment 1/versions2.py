# Load libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Load dataset 
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataset = pd.read_csv(url, names=names)


'''Evaluation of Algorithms and Prediction'''
# Split-out validation dataset
array = dataset.values
X = array[:,0:8]
y = array[:,8]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

#Spot Check algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

#Evaluate each algorithm in turn
results=[]
names2=[]
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names2.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
plt.boxplot(results, labels=names2)
plt.title('Algorithm Comparison')
plt.show()


'''Visualizing the data'''
#Histogram plots
dataset.hist()
plt.show()

#Density plots
dataset.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
plt.show()

#Box and Whisker plots
dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False)
plt.show()

#Correlation Matrix plot
correlations = dataset.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

#Scatterplot Matrix
scatter_matrix(dataset)
plt.show()

#Line Plot
'''comment: I was not sure what axis to plot the variable, and I could not quite make sense 
or interpretation of the line plot so I plotted both axes, then one with default X-points
'''
bloodpres = np.array(dataset.pres.to_numpy())
plt.plot(np.sort(bloodpres), y)
plt.show()
plt.plot(y, np.sort(bloodpres))
plt.show()
plt.plot(np.sort(bloodpres))
plt.show()

#Bar plot
glucse = dataset.plas.to_numpy()
x_gluc = ["(0-100)" , "(101-150)" , "(151-200)"]
y_gluc = [np.count_nonzero(glucse<=100), np.count_nonzero((100<glucse) & (glucse<=150)), np.count_nonzero(glucse>150)]
plt.bar(x_gluc,y_gluc)
plt.show()

#Scatter plot
BMI = dataset.mass
Insulin = dataset.test
plt.scatter(BMI, Insulin)
plt.show()
