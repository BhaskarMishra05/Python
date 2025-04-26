from sklearn import datasets
import numpy as np
iris=datasets.load_iris()
from sklearn.model_selection import train_test_split
X= iris.data 
y=iris.target 
num=X.select_dtypes(include=['int64','float64']).columns.tolist()
cat=X.select_dtypes(include=['object']).columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print("Training x data: ",X_train.shape)
print("Testing set size: ",X_test.shape) 

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
pre=ColumnTransformer([('scaler',StandardScaler(),num),
                       ('encoder',OneHotEncoder(drop='first'),cat)])
model=Pipeline([('pred',pre),('lin',LogisticRegression())])
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Predict label: ",y_pred)
print("Actual label: ",y_test)


print()
print("Check accuracy")
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(f"Model accuracy: {accuracy* 100:.2f} %")
print()
