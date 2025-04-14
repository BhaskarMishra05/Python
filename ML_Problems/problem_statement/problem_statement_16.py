'''🧠 Problem Statement: Classify Handwritten Digits (MNIST Subset)
You're given a dataset of handwritten digits (0–9) from the sklearn.datasets.load_digits() dataset.
Your task is to classify the digit based on the image data using a classification model, 
and display the actual vs predicted values along with the digit names (as strings).'''
from sklearn import datasets
digit=datasets.load_digits()
x=digit.data 
y=digit.target 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
model=Pipeline([('scaler',StandardScaler()),
                ('knn',KNeighborsClassifier())])

model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score
print(f'Accuracy score: {accuracy_score(y_test,y_pred)*100 :.2f}')
for actual,predict in zip(y_test[:10],y_pred[:10]):
    print(f'Actual: {actual}  | Predicted: {predict}')
import matplotlib.pyplot as plt
plt.scatter('y_test','y_pred')
plt.show()