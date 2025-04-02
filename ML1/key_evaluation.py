from sklearn import datasets
dataset=datasets.load_iris()
from sklearn.model_selection import train_test_split
x=dataset.data 
y=dataset.target
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=4)
model.fit(x_train,y_train)
pred=model.predict(x_test)
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
acc=accuracy_score(y_test,pred)
precision=precision_score(y_test,pred,average='weighted')
recall=recall_score(y_test,pred,average='weighted')
f1=f1_score(y_test,pred,average='weighted')
print(f"Accuracy score: {acc}")
print(f"Precision score: {precision}")
print(f"Recall score: {recall}")
print(f"F1 score: {f1}")
from sklearn.metrics import classification_report
report=classification_report(y_test,pred)
print()
print(f"Classification report: {report}")