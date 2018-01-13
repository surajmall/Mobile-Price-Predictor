import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import Series, DataFrame
from sklearn import datasets, svm, tree
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

df = pd.read_csv('train.csv')
num_rows = df.shape[0]
y = df['price_range']
del df['price_range']

#Computation for missing values

counter_nan = df.isnull().sum()
counter_without_nan = counter_nan[counter_nan==0]
df = df[counter_without_nan.keys()]

x = df.ix[:,:-1].values
standard_scaler = StandardScaler()
x_std = standard_scaler.fit_transform(x)
tsne = TSNE(n_components=3, random_state=0)
x_test_2d = tsne.fit_transform(x_std)

#Map Plotting

markers = ('s','d','o','^')
color_map = {0:'red', 1:'blue', 2:'lightgreen', 3:'purple', 4:'cyan'}
plt.figure()
for idx, cl in enumerate(np.unique(x_test_2d)):
    plt.scatter(x=x_test_2d[cl, 0],y=x_test_2d[cl,1], c=color_map[idx], marker=markers[idx], label=cl)
plt.show()

plt.figure(figsize=(10000,80000))
#plt = sns.heatmap(df.corr(),annot=True,cmap='cubehelix_r')
#plt.imshow(df, cmap='hot', interpolation='nearest')
plt.show()

#X_train, X_test, y_train, y_test = train_test_split(x_test_2d,y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(x_std,y, test_size=0.2)
#clf=AdaBoostClassifier(RandomForestClassifier(),n_estimators=10,learning_rate=1.0)
#clf=LineaRegression()

svc = svm.SVC()
parameters = {'kernel':('linear','rbf'), 'C':[1,10]}
clf=GridSearchCV(svc,parameters,cv=10)
model = clf.fit(X_train,y_train)

pred=clf.predict(X_test)
plt.scatter(y_test, pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")

print ("Score:", model.score(X_test, y_test))

#Output file calculations
dfd = pd.read_csv('test.csv')
y = dfd.ix[:,:-1].values
standard_scaler1 = StandardScaler()
x_std1 = standard_scaler1.fit_transform(y)

prediction = clf.predict(x_std1)
#np.savetxt('./GridSearchCV.csv',prediction, delimiter=',')'''
