'''📦 Problem: Predict Loan Default
A bank wants to predict whether a customer will default on a loan based on certain financial and behavioral indicators.

🔢 Features:
income — Monthly income of the customer (in USD)
credit_score — Score between 300 to 850
loan_amount — Loan amount requested (in USD)
loan_term_months — Duration of the loan in months
employment_status — Employment status (Employed / Unemployed / Self-Employed)
owns_house — Does the customer own a house? (Yes/No)
defaulted — 🎯 Target: Did the customer default? (Yes/No) '''

import numpy as np
import pandas as pd
np.random.seed(42)
n=500 
data=({'income':np.random.uniform(500,5000,n),
'credit_score': np.random.uniform(300,850,n),
'loan_amount': np.random.uniform(500,15000,n),
'loan_term_months': np.random.randint(4,24,n),
'employment_status': np.random.choice(['Employed','Unemployed','Self-Employed'],n),
'owns_house': np.random.choice(['Yes','No'],n),
'defaulted':  np.random.choice(['Yes','No'],n)})

df=pd.DataFrame(data)
x=df.drop('defaulted',axis=1)
y=df['defaulted']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
num_cols=x.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols=x.select_dtypes(include=['object']).columns.tolist()

preprocessing=ColumnTransformer([('scaler',StandardScaler(),num_cols),
                                 ('encoder',OneHotEncoder(drop='first'),cat_cols)])

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
models=({'DecisionTreeClassifier':DecisionTreeClassifier(),
         'RandomForestClassifier':RandomForestClassifier(),
         'ExtraTreesClassifier':ExtraTreesClassifier(),
         'GradientBoostingClassifier':GradientBoostingClassifier(),
         'SVC':SVC()})

results={}
for name,model in models.items():
    pipe=Pipeline([('pre',preprocessing),
                   ('model',model)])
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    results[name]=acc
    print(f'Accuracy for {name} : {acc}')