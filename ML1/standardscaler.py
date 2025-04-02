from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
iris=datasets.load_iris()
x=iris.data 
y=iris.target 
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.89)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
from sklearn.tree import DecisionTreeClassifier
m=DecisionTreeClassifier()
m.fit(x_train_scaled,y_train)
pred=m.predict(x_test_scaled)
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,pred)
print(f"accuracy : {acc*100 :.2f}")
