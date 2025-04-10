'''🏥 Real-World Healthcare Problem: Predict Diabetes Onset
🔬 Problem Statement:
You are provided with patient data including medical measurements and lifestyle indicators. Your goal is to predict whether a patient is likely to develop diabetes.

📦 Features:
age – Age of the patient

bmi – Body Mass Index

blood_pressure – Systolic BP

glucose_level – Fasting blood glucose level (mg/dL)

insulin_level – Insulin level (μU/mL)

physical_activity – Is the patient physically active? (Yes / No)

smoking – Is the patient a smoker? (Yes / No)

🎯 Target:
diabetes – Whether the patient is diabetic (Yes / No)

'''
import numpy as np
import pandas as pd
np.random.seed(42)
n_sample=500
age = np.random.uniform(18,80,n_sample)
bmi = np.random.uniform(14.5,40,n_sample)
blood_pressure = np.random.uniform(70,180,n_sample)
glucose_level = np.random.uniform(70,200,n_sample)
insulin_level = np.random.uniform(2,300,n_sample)
physical_activity = np.random.choice(['Yes','No'],n_sample)
smoking = np.random.choice(['Yes','No'],n_sample)
diabetes = np.random.choice(['Yes','No'],n_sample, p=[0.3, 0.7]) # p=[0.4, 0.6] imporove acc by 5%
df=pd.DataFrame({'age':age,'bmi':bmi,'bp':blood_pressure,'gl':glucose_level,'il':insulin_level,'pa':physical_activity,
                 'smoking':smoking,'diabetes':diabetes})
x=df.drop('diabetes',axis=1)
y=df['diabetes']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
pre=ColumnTransformer([('scaler',StandardScaler(),['age','bmi','bp','gl','il']),
                       ('encoder',OneHotEncoder(drop='first'),['pa','smoking'])])
model=Pipeline([('pre',pre),
                ('rain',RandomForestClassifier())])
model.fit(x_train,y_train)
prediction=model.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report
print(f'Accuracy score: {accuracy_score(y_test,prediction)*100 :.2f}')
print(f'Report : {classification_report}')
for actual,pred in zip(y_test[:10],prediction[:10]):
    print(f'Actaul {actual}     | Predicted {pred}')