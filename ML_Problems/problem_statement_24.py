import numpy as np
import pandas as pd
np.random.seed(42)
n = 300
subscription_length = np.random.randint(1, 36, n)
monthly_usage = np.random.uniform(5, 100, n)
num_devices = np.random.randint(1, 6, n)
support_tickets = np.random.poisson(1.5, n)
premium_user = np.random.choice(['Yes', 'No'], n, p=[0.4, 0.6])
churn = np.random.choice(['Yes', 'No','Maybe','Dont Know'], n, p=[0.25,0.25,0.25,0.25])
df = pd.DataFrame({
    'subscription_length': subscription_length,
    'monthly_usage': monthly_usage,
    'num_devices': num_devices,
    'support_tickets': support_tickets,
    'premium_user': premium_user,
    'churn': churn
})
x=df.drop('churn',axis=1)
y=df['churn']
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
preprocess=ColumnTransformer([('scaler',StandardScaler(),['subscription_length'
                                                          ,'monthly_usage','num_devices','support_tickets']),
                              ('encoder',OneHotEncoder(drop='first'),['premium_user'])])
model=Pipeline([('pre',preprocess),('rfr',RandomForestClassifier())])
model.fit(x_train,y_train)
prediction=model.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(f'Accuracy: {accuracy_score(y_test, prediction)*100:.2f}%')
print("Confusion Matrix:\n", confusion_matrix(y_test, prediction))
print("Classification Report:\n", classification_report(y_test, prediction))
for actual,predicted in zip(y_test[:10],prediction[:10]):
    print(f'Actual: {actual}    | Predicted: {predicted}')