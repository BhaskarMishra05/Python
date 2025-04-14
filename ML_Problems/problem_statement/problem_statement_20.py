'''🧪 🧠 Problem Statement: Predict Disease Based on Symptoms
You're working with a synthetic dataset of patient symptoms and vitals.
Your task is to build a classification model to predict the disease a patient is likely to have.

📦 Dataset Features:
fever – Yes or No

cough – Yes or No

headache – Yes or No

age – Age of the patient

bp_level – Blood pressure level (Low, Normal, High)

disease (🎯 Target):

Flu

Hypertension

Migraine

None

'''

import numpy as np
import pandas as pd
np.random.seed(42)
n_sample=100

fever = np.random.choice(['Yes','No'],n_sample)
cough = np.random.choice(['Yes','No'],n_sample)
headache = np.random.choice(['Yes','No'],n_sample)
age = np.random.uniform(18,80,n_sample)
bp_level = np.random.choice(['High','Normal','Low'],n_sample)
disease = np.random.choice(['Flu','Hypertension','Migraine','None'],n_sample)
df=pd.DataFrame({'fever':fever,'cough':cough,'headache':headache,'age':age,'bp':bp_level,
    'disease':disease})
x=df[['fever','cough','headache','age','bp']]
y=df['disease']
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
preprocess=ColumnTransformer([('scaler',StandardScaler(),['age']),
                              ('encoder',OneHotEncoder(drop='first'),['fever','cough','headache','bp'])])
model=Pipeline([('preprocess',preprocess),
                ('linear',DecisionTreeClassifier())])
model.fit(x_train,y_train)
model_prediction=model.predict(x_test)
print(f'Accuracy score : {accuracy_score(y_test,model_prediction)*100 :.2f}')
 # print(f'Confusion Metrix : {confusion_matrix(y_test,model_prediction)}')
 # print(f'Classification : {classification_report(y_test,model_prediction)}')
for actual,predict in zip(y_test[:10],model_prediction[:10]):
    print(f'Actual : {actual}   | Predicted: {predict}')