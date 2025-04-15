from sklearn import datasets
iris=datasets.load_iris()
print("Keys of iris dataset: ",iris.keys())
print()
print(iris.target_names)
print(iris.feature_names)
print(iris.data[:5])
print(iris.target[:5])

print("Test_+train_split")
from sklearn.model_selection import train_test_split
X= iris.data 
y=iris.target 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print("Training x data: ",X_train.shape)
print("Testing set size: ",X_test.shape) 

print("Import Decision tree classifers: ")
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(max_depth=2)
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
species_mapping={0: "Setosa", 1: "Versicolor", 2: "Virginica"}
for actual,pred in zip(y_test[:100],y_pred[:100]):
    actual_label=species_mapping[actual]
    pred_label=species_mapping[pred]
    print("Actual: " , actual_label , "Predicted: " ,pred_label)
    
print("\nEnter new data for prediction (sepal length, sepal width, petal length, petal width):")
new_data = [float(input("Sepal Length: ")), float(input("Sepal Width: ")),
            float(input("Petal Length: ")), float(input("Petal Width: "))]
new_data = [new_data]  # Wrapping the input data as a list of lists to match the input format

# Make prediction
new_prediction = model.predict(new_data)
print(f"Predicted Species: {species_mapping[new_prediction[0]]}")