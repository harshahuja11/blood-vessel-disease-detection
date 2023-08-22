import pandas as pd
import matplotlib.pylab as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('/Users/DELL/Desktop/AI project/dataset.csv')
# Separate the features from the response
y = data["target"].copy()
X = data.drop("target", axis=1)
# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Logistic Regression
l_reg = LogisticRegression(max_iter=1000, random_state=42)
l_reg.fit(X_train, y_train)
l_reg_predict_train = l_reg.predict(X_train)
l_reg_acc_score_train = accuracy_score(y_train, l_reg_predict_train)
print("Train Accuracy of Logistic Regression=", l_reg_acc_score_train, '\n')
l_reg_predict_test = l_reg.predict(X_test)
l_reg_acc_score_test = accuracy_score(y_test, l_reg_predict_test)
print("Test Accuracy of Logistic Regression=", l_reg_acc_score_test, '\n')
l_reg_conf_matrix = confusion_matrix(y_test, l_reg_predict_test)
print("confusion matrix")
print(l_reg_conf_matrix,'\n')
# Random Forest
rf = RandomForestClassifier(n_estimators=39,max_depth=9,random_state=42)
rf.fit(X_train,y_train)
rf_predicted_train = rf.predict(X_train)
rf_acc_score_train = accuracy_score(y_train, rf_predicted_train)
print("Train Accuracy of Random Forest=",rf_acc_score_train,'\n')
rf_predicted_test = rf.predict(X_test)
rf_acc_score_test = accuracy_score(y_test, rf_predicted_test)
print("Test Accuracy of Random Forest=",rf_acc_score_test,'\n')
rf_conf_matrix = confusion_matrix(y_test, rf_predicted_test)
print("confusion matrix")
print(rf_conf_matrix,'\n')
# Bar Plot for Accuracy
fig = plt.figure(figsize=(8, 6))
Algorithm = ['LR','RF']
accuracy = [79.5,98.5,100,98.5,98.5]
plt.bar(Algorithm,accuracy,color='b', width=0.5)
plt.ylabel('Accuracy (percent)')
plt.xlabel('Algorithm')
plt.title('Accuracy comparison')
plt.show()