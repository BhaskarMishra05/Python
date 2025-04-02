from sklearn import datasets
iris=datasets.load_iris()
from sklearn.model_selection import train_test_split
x=iris.data 
y=iris.target 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.tree import DecisionTreeClassifier
m1=DecisionTreeClassifier(max_depth=2)
m1.fit(x_train,y_train)
pred1=m1.predict(x_test)
from sklearn.neighbors import KNeighborsClassifier
m2=KNeighborsClassifier(n_neighbors=3)
m2.fit(x_train,y_train)
pred2=m2.predict(x_test)
from sklearn.metrics import accuracy_score
acc1=accuracy_score(y_test,pred1)
acc2=accuracy_score(y_test,pred2)
print(f"Accuracy score of Decision tree: {acc1*100}")
print(f"Accuracy score of KNN: {acc2*100}")
