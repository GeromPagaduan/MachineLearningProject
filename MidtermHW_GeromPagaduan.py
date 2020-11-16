import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder



iris = np.array(pd.read_csv('iris.csv'))
# print(iris.head())
# sns.pairplot(data=iris, hue='variety', palette='Set2')
# plt.show()
label = iris[:, -1] #data (rows - Y), the type of flower
data = iris[:, :-1] #label (columns - X), the features || NOTE: use 2:-1 to use the last 2 columns
# print(data)

#transforms labels to integers
le = LabelEncoder()
ylabels = le.fit_transform(label)

# Partition data into 70% training, 10% validation, 20% testing
X_train,X_test,Y_train,Y_test=train_test_split(data,ylabels,train_size=0.7, test_size=0.3,random_state=0)
X_test, X_valid, y_test, y_valid = train_test_split(data, ylabels, test_size=0.66, random_state=0)


# Logistic Regression
print("Logistic Regression")
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y_lr_pred = logreg.predict(X_test)
print(classification_report(y_test, y_lr_pred))
#cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
#print(cnf_matrix)


# K Nearest Neighbors
print("K Nearest Neighbors")
K = [3, 5, 7]
type = [1, 2]
best_k = 0
best_type = 0
best_acc = 0
for type_value in type:
    for k_value in K:
        model = KNeighborsClassifier(n_neighbors=k_value, p=type_value)
        model.fit(X_train, Y_train)
        y_knn_pred = model.predict(X_valid)
        tested_acc = metrics.accuracy_score(y_valid, y_knn_pred) # gets accuracy

        # stores best accuracy
        if tested_acc > best_acc:
            best_acc = tested_acc
            best_k = k_value
            best_type = type_value
model = KNeighborsClassifier(n_neighbors=best_k, p=best_type)
model.fit(X_train, Y_train)
print(classification_report(y_test, model.predict(X_test), target_names=le.classes_))


# Support Vector Machine
print("Support Vector Machine")
Gamma = 0.001
C = 1
model = SVC(kernel='linear', C=C, gamma=Gamma)
model.fit(X_train, Y_train)
svm_pred = model.predict(X_test)
#print(confusion_matrix(y_test, pred))
print(classification_report(y_test, svm_pred))