from sklearn import datasets
wine=datasets.load_wine()
x=wine.data 
y=wine.target 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
model=Pipeline([('scaler',StandardScaler()),
                ('knn',KNeighborsClassifier(n_neighbors=5))])
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score
print(f'Accuracy score: {accuracy_score(y_test,y_pred) } %')
for actual,prediction in zip(y_test,y_pred):
    print(f'Actual: {actual}  | Predicted: {prediction}')
