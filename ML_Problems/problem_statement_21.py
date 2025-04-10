'''🔬 🧠 Problem Statement: Predict Disease Severity Based on Patient Symptoms
You're given a synthetic dataset simulating various patient symptoms and medical observations.
Your task is to predict the severity of the disease using a classification model.

📦 Dataset Features:
temperature — Body temperature in Celsius (e.g. 36.5–41.0)

heart_rate — Heart rate (beats per minute)

fatigue — Is the patient experiencing fatigue? (Yes/No)

shortness_of_breath — Difficulty in breathing? (Yes/No)

oxygen_saturation — Blood oxygen levels (percentage)

severity (🎯 Target):

Mild

Moderate

Severe'''
import numpy as np
import pandas as pd
np.random.seed(42)
n_sample=100
temperature = np.random.uniform(36.5,41.0,n_sample)
heart_rate = np.random.uniform(50,200,n_sample)
fatigue = np.random.choice(['Yes','No'],n_sample)
shortness_of_breath = np.random.choice(['Yes','No'],n_sample)
oxygen_saturation = np.random.uniform(1,100,n_sample)
severity = np.random.choice(['Mild','Moderate','Severe'],n_sample)
df=pd.DataFrame({'temp':temperature,'hr':heart_rate,'f':fatigue,'sob':shortness_of_breath,'os':oxygen_saturation,'s':severity})
x=df[['temp','hr','sob','os','f']]
y=df['s']
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.metrics import accuracy_score
pre=ColumnTransformer([('scaler',StandardScaler(),['temp','hr','os']),
                       ('encode',OneHotEncoder(drop='first'),['f','sob'])])
model=Pipeline([('pre',pre),('tree',DecisionTreeClassifier())])
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(f'acc score: {accuracy_score(y_test,y_pred)*100 :.2f}')
for a,b in zip(y_test[:10],y_pred[:10]):
    print(f'Actual: {a}    | predicted: {b}')





