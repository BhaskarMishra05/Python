'''🧠 Problem: Predict Employee Attrition (Yes/No)
A company wants to predict whether an employee is likely to leave the company based on a few basic features.

📦 Features:
experience_years: Years of work experience

monthly_income: Monthly salary in thousands

job_satisfaction: Level from 1 (low) to 5 (high)

overtime: Works overtime? (Yes/No)

remote: Works remotely? (Yes/No)

attrition: 🎯 Target (Yes/No)

'''
import numpy as np
import pandas as pd
np.random.seed(42)
n=200
experience_years = np.random.uniform(0,25,n)
monthly_income= np.random.uniform(100,2500,n)
job_satisfaction= np.random.uniform(1,5,n)
overtime = np.random.choice(['Yes',"No"],n,p=[0.5,0.5])
remote= np.random.choice(['Yes',"No"],n,p=[0.5,0.5])
attrition= np.random.choice(['Yes',"No"],n,p=[0.5,0.5])
df=pd.DataFrame({'exp':experience_years,'mi':monthly_income,'js':job_satisfaction,'overtime':overtime,
                 'remote':remote,'target':attrition})
x=df.drop('target',axis=1)
y=df['target']
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,r2_score
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
preprocess=ColumnTransformer([('scaler',StandardScaler(),['exp','mi','js']),
                              ('encoder',OneHotEncoder(drop='first'),['overtime','remote'])])
model=Pipeline([('pre',preprocess),('rfc',RandomForestClassifier())])
model.fit(x_train,y_train)
prediction=model.predict(x_test)
print(f'Accuracy Score: {accuracy_score(y_test,prediction) *100 :.2f}')
for actual,predicted in zip(y_test[:10],prediction[:10]):
    print(f'Actual: {actual}    | Predicted: {predicted}')