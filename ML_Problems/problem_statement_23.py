import numpy as np
import pandas as pd
np.random.seed(42)
n = 30000
age = np.random.randint(20, 90, n)
days_in_hospital = np.random.randint(1, 15, n)
num_lab_procedures = np.random.randint(5, 50, n)
num_medications = np.random.randint(1, 20, n)
has_insurance = np.random.choice(['Yes', 'No'], n, p=[0.7, 0.3])
comorbidities = np.random.randint(0, 5, n)
discharged_to_home = np.random.choice(['Yes', 'No'], n, p=[0.8, 0.2])
readmitted = np.random.choice(['Yes', 'No'], n, p=[0.50, 0.50])

df = pd.DataFrame({
    'age': age,
    'days_in_hospital': days_in_hospital,
    'num_lab_procedures': num_lab_procedures,
    'num_medications': num_medications,
    'has_insurance': has_insurance,
    'comorbidities': comorbidities,
    'discharged_to_home': discharged_to_home,
    'readmitted': readmitted })
x=df.drop('readmitted',axis=1)
y=df['readmitted']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'days_in_hospital', 'num_lab_procedures', 'num_medications', 'comorbidities']),
    ('cat', OneHotEncoder(drop='first'), ['has_insurance', 'discharged_to_home'])
])
model = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(random_state=42))
])
model.fit(x_train,y_train)
prediction=model.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report
print(f'Accuracy score: {accuracy_score(y_test,prediction)*100 :.2f}')
print(f'Report: {classification_report(y_test,prediction)}')
for a,b in zip(y_test[:10],prediction[:10]):
    print(f'Acutal: {a}    | Predicted: {b}')