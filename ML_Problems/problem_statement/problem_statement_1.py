# You are given the Breast Cancer dataset, which contains features about tumors. 
# Your task is to train a machine learning model to 
# classify whether a tumor is malignant (cancerous) or benign (non-cancerous).
from sklearn import datasets
cancer=datasets.load_breast_cancer()
print(cancer.keys())
print(cancer.feature_names)
print()
print(cancer.target_names)
print()
x=cancer.data
y=cancer.target
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
model.fit(x_train_scaled,y_train)
prediction_model=model.predict(x_test_scaled)
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix
accuracy=accuracy_score(y_test,prediction_model)
precision=precision_score(y_test,prediction_model,average='binary')
f1=f1_score(y_test,prediction_model,average='binary')
recall=recall_score(y_test,prediction_model,average='binary')
matrix=confusion_matrix(y_test,prediction_model)
print(f"Accuracy score: {accuracy*100 :.2f}")
print(f"Precision score: {precision*100 :.2f}")
print(f"F1 score: {f1*100 :.2f}")
print(f"Recall score: {recall*100 :.2f}")
for actual,predicted in zip(y_test[:10],prediction_model[:10]):
    if actual==0:
        actual_label="Malignant"
    else:
        actual_label="Benign"
    if predicted==0:
        predicted_label="Malignant"
    else:
        predicted_label="Benign"
    print("Actual:  ",actual_label ,"     Predicted : " ,predicted_label)