from sklearn import datasets
breast_cancer =datasets.load_breast_cancer()
from sklearn.model_selection import train_test_split
x=breast_cancer.data 
y=breast_cancer.target 
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.345)
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=4)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
model.fit(x_train_scaled,y_train)
prediction=model.predict(x_test_scaled)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,prediction)
print(f"Accuracy of Breast cancer model using KNN is {accuracy*100 :.2f}%")
from sklearn.tree import DecisionTreeClassifier
tree_model=DecisionTreeClassifier(max_depth=4)
tree_model.fit(x_train_scaled,y_train)
prediction_tree=tree_model.predict(x_test_scaled)
accuracy_tree=accuracy_score(y_test,prediction_tree)
print(f"Accuracy of Decision tree is : {accuracy_tree*100 :.2f}%")