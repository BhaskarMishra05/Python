'''Problem Statement
You are given the Iris Dataset 🌸 — a classic dataset that includes
features of 3 different types of flowers: Setosa, Versicolor, and Virginica.

Your goal is to:

Classify the flower type based on its features.

Evaluate the model using metrics.

Implement the solution once without a pipeline, and then again using a pipeline.'''
from sklearn import datasets
iris=datasets.load_iris()
from sklearn.model_selection import train_test_split
x=iris.data
y=iris.target 

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
model_without_pipeline=KNeighborsClassifier()
model_without_pipeline.fit(x_train_scaled,y_train)
model_prediction_without_pipeline=model_without_pipeline.predict(x_test_scaled)
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,model_prediction_without_pipeline)
print(f'Accuracy without Pipeline: {acc *100 :.2f}')

    
    
from sklearn.pipeline import Pipeline
model_pipeline=Pipeline([('scaler',StandardScaler()),
                 ('regressor',KNeighborsClassifier())])
model_pipeline.fit(x_train,y_train)
model_prediction_with_pipeline=model_pipeline.predict(x_test)
acc_1=accuracy_score(y_test,model_prediction_with_pipeline)
print(f'Accuracy with pipeline: {acc_1*100 :.2f}')

for actual, predicted in zip(y_test[:10], model_prediction_without_pipeline[:10]):
    actual_name = iris.target_names[actual]
    predicted_name = iris.target_names[predicted]
    print(f'Actual: {actual_name:<10} | Predicted: {predicted_name:<10}')


