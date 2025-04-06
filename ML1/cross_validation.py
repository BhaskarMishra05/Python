import numpy as np
from sklearn import datasets
vine=datasets.load_wine()
x=vine.data 
y=vine.target 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
model=Pipeline([('scaler',StandardScaler()),
                ('regression',LogisticRegression())])
cv_score=cross_val_score(model,x,y,cv=10)

print(f'Cross validatin: {cv_score },')
print(f'Accuracy score: {np.mean(cv_score)*100 :.2f}%')