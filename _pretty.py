from prettytable 
import PrettyTable
table = PrettyTable()
table.field_names = ["Model","Accuracy", "Mean Squared Error", "R² score","Mean Absolute Error"]
models = [
    LogisticRegression(),
    KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski'),
    SVC(kernel='linear',random_state=0),
    GaussianNB(),
    DT(criterion='entropy', random_state=0),
    RF(n_estimators=10, criterion='entropy', random_state=0),
    Perceptron(tol=1e-3, random_state=0)
]
for model in models:
    model.fit(XTrain, yTrain) 
    yPred = model.predict(XTest)
    accuracy = accuracy_score(yTest,yPred)
    mse = mean_squared_error(yTest,yPred)
    r = r2_score(yTest,yPred)
    mae = mean_absolute_error(yTest,yPred)
table.add_row([type(model).__name__, format(accuracy, '.3f'),format(mse, '.3f'),format(r, '.3f'),format(mae, '.3f')])
    
table.add_row(["Artificial Neural Network Classifier",0.954 ,0.045,0.817,0.045])
print(table)
