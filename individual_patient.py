import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# for model improvement
from sklearn.ensemble import VotingClassifier

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(solver='lbfgs', max_iter=1000)
import joblib
heart_data = pd.read_csv("dataset.csv", encoding='ANSI')
heart_data.head(10)
heart_data['target'].value_counts()

sns.countplot(x=heart_data["target"])
sns.pairplot(heart_data, hue= 'target',vars = ['age', 'sex', 'cp', 'trestbps', 'chol' ])
plt.figure(figsize= (16,9))
sns.heatmap(heart_data.corr(), annot = True, cmap='coolwarm', linewidths = 2)
X = heart_data.drop(columns = 'target', axis = 1)

Y = heart_data['target']
Y.head()
scaler = StandardScaler()
scaler.fit(X)
X_standard = scaler.transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, stratify = Y, random_state = 3 )

print(X.shape, X_train.shape, X_test.shape)
# instantiate the model
model=LogisticRegression(max_iter=3000)
# training the LogisticRegression model with training data
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(Y_test, y_pred)))

# instantiate the model
gnb = GaussianNB()
# model = gnb

# fit the model
gnb.fit(X_train, Y_train)
y_pred = gnb.predict(X_test)

y_pred
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(Y_test, y_pred)))

# instantiate the model
knn = KNeighborsClassifier(n_neighbors=7)


# fit the model
knn.fit(X_train, Y_train)

y_pred = knn.predict(X_test)

y_pred
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(Y_test, y_pred)))

# Create Decision Tree classifer object
dtc = DecisionTreeClassifier()


# fit the model
dtc.fit(X_train,Y_train)

y_pred = dtc.predict(X_test)

y_pred
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(Y_test, y_pred)))

# overfitted

# instantiate the model
svm = SVC(kernel='linear')

# fitting x samples and y classes
svm.fit(X_train, Y_train)

y_pred = svm.predict(X_test)

y_pred
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(Y_test, y_pred)))

svc = SVC(kernel = 'sigmoid', gamma = 1.0) # A higher gamma value means that each training example will have a greater influence on the decision boundary.
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth = 5)
lrc = LogisticRegression(solver = 'liblinear', penalty = 'l1') # liblinear is parameter specifies the solver to use,
# L1 penalty is a type of regularization that helps to prevent overfitting.

rfc = RandomForestClassifier(n_estimators= 50, random_state = 2)  # n_estimators : the number of trees in the forest,
# random_state : specifies the random seed that is used to initialize the random forest

abc = AdaBoostClassifier(n_estimators = 50, random_state = 2)
bc = BaggingClassifier(n_estimators = 50, random_state = 2)
etc = ExtraTreesClassifier(n_estimators = 50, random_state = 2)
gbdt = GradientBoostingClassifier(n_estimators = 50, random_state = 2)
xgb = XGBClassifier(n_estimators = 50, random_state=2)

classification = {
    'Support Vector Classifier': svc,
    'K-Neighbors Classifier': knc,
    'Multinomial NB': mnb,
    'Decision Tree Classifier': dtc,
    'Logistic Regression': lrc,
    'Random Forest Classifier': rfc,
    'AdaBoost Classifier': abc,
    'Bagging Classifier': bc,
    'Extra Trees Classifier': etc,
    'Gradient Boosting Classifier': gbdt,
    'XGB Classifier': xgb
}

def train_classifier(classification, X_train, y_train, X_test, y_test):
  classification.fit(X_train, y_train)
  y_pred = classification.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  matrix = confusion_matrix(y_test, y_pred)

  return accuracy, precision, matrix

accuracy_scores = []
precision_scores = []

for name, cls in classification.items():
  curr_accuracy, curr_precision, matrix = train_classifier(cls, X_train, Y_train, X_test, Y_test)
  print("Model name : ", name)
  print("Accuracy : ", curr_accuracy)
  print("Precision : ", curr_precision)
  print("Confusin-Matrix : ", matrix, '\n')

  accuracy_scores.append(curr_accuracy)
  precision_scores.append(curr_precision)

  result_dataframe = pd.DataFrame.from_dict(
      {'Algorithm': classification.keys(), 'Accuracy': accuracy_scores, 'Precision': precision_scores}, orient='index')
  result_dataframe = result_dataframe.transpose()

  print(result_dataframe)  # Print all of them out here
  rfc = RandomForestClassifier(n_estimators=50, random_state=2)
  bc = BaggingClassifier(n_estimators=50, random_state=2)
  etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
  xgb = XGBClassifier(n_estimators=50, random_state=2)
  voting = VotingClassifier(estimators=[('rfc', rfc), ('bc', bc), ('et', etc), ('xgb', xgb)], voting='soft')
voting.fit(X_train, Y_train)

y_pred = voting.predict(X_test)

print(accuracy_score(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))
print(precision_score(Y_test, y_pred))

# voting model is most accurate and precise

# accuracy of traning data
# accuracy function measures accuracy between two values,or columns

X_train_prediction = voting.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("The accuracy of training data : ", training_data_accuracy)

Y_pred = voting.predict(X_test)


accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy   :", accuracy)
precision = precision_score(Y_test, Y_pred)
print("Precision  :", precision)
recall = recall_score(Y_test, Y_pred)
print("Recall     :", recall)
F1_score = f1_score(Y_test, Y_pred)
print("F1-score   :", F1_score)

# check results
print(metrics.classification_report(Y_test, Y_pred))

# confusion matrix

cm = confusion_matrix(Y_test,Y_pred)

#Plot the confusion matrix.
sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=['malignant', 'benign'],
            yticklabels=['malignant', 'benign'])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
# plt.show()
print("Enter Input data : ")

lst = []

# number of elements as input
age = int(input("Age : "))
lst.append(age)
sex = int(input("Sex : "))
lst.append(sex)
chest_pain_type = int(input("Chest Pain : "))
lst.append(chest_pain_type)
resting_bp = int(input("Resting BP : "))
lst.append(resting_bp)
serum_chl = int(input("Serum Cholestoral : "))
lst.append(serum_chl)
fasting_bs = int(input("Fasting blood sugar : "))
lst.append(fasting_bs)
rest_el = int(input("Resting electrocardiographic results : "))
lst.append(rest_el)
max_hr = int(input("Maximum heart rate achieved : "))
lst.append(max_hr)
exc_inc = int(input("Exercise induced angina : "))
lst.append(exc_inc)
oldpeak = int(input("Oldpeak : "))
lst.append(oldpeak)
slope_peak = int(input("Slope of the peak exercise ST segment  : "))
lst.append(slope_peak)
blood_vess = int(input("Number of major vessels (0-3)  : "))
lst.append(blood_vess)
chest_type = int(input("chest pain type : "))
lst.append(chest_type)


# input_data = (58,0,3,150,283,1,0,162,0,1,2,0,2)

# changing data to numpy array
input_data_array = np.asarray(lst)

# reshape the array as we are predicting for one instance
input_data_reshaped =  input_data_array.reshape(1,-1)

# standarize the input data
# std_data = scaler.transform(input_data_reshaped)
# print(std_data[0])
# predicting the result and printing it

prediction = voting.predict(input_data_reshaped)

print(prediction)

if(prediction[0] == 0):
    print("Patient has a healthy heart 💛💛💛💛")

else:
    print("Patient has a heart Disease 💔💔💔💔")


pickle.dump(voting, open("trained_model.pkl", 'wb'))
# saving file
oaded_model = pickle.load(open("trained_model.pkl",'rb'))
# save the model to disk
filename = 'heart_model.sav'
joblib.dump(voting, filename)


