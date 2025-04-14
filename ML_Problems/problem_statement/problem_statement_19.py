'''🏥 Problem Statement: Predict Patient Risk Category
You are provided with a synthetic dataset representing medical data of patients.
Your task is to classify patients into risk categories — Low, Medium, or High — based on their health metrics.

📦 Dataset Features:
age — Age of the patient

bmi — Body Mass Index

bp — Blood Pressure (systolic)

cholesterol — Cholesterol Level (in mg/dL)

risk_category (🎯 Target):

Low

Medium

High

'''
import numpy as np
import pandas as pd
np.random.seed(42)

n_samples=100
age = np.random.randint(20, 80, n_samples)
bmi = np.random.uniform(18, 40, n_samples)
bp = np.random.randint(90, 180, n_samples)
cholesterol = np.random.randint(150, 300, n_samples)
risk_category = np.random.choice(['Low', 'Medium', 'High'], n_samples)

df=pd.DataFrame({'age':age,'bmi':bmi,'bp':bp,'cholesterol':cholesterol,'risk':risk_category})
x=df[['age','bmi','bp','cholesterol']]
y=df['risk']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
# from sklearn.pipeline import Pipeline lets not use pipeline this time
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaler=scaler.fit_transform(x_train)
x_test_scaler=scaler.transform(x_test)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train_scaler,y_train)
y_pred=model.predict(x_test_scaler)
from sklearn.metrics import accuracy_score
print(f'Accuracy score: {accuracy_score(y_test,y_pred) * 100 :.2f}')
for actual,predict in zip(y_test[:10],y_pred[:10]):
    print(f'Actual : {actual}   |  Predicted: {predict}')