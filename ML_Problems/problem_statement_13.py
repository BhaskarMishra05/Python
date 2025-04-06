import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import datasets
iris=datasets.load_iris()
x=iris.data 
y=iris.target 
models={'tree':DecisionTreeClassifier(),
        'KNN':KNeighborsClassifier(),
        'linear':LogisticRegression()}
for names,model in models.items():
    pipeline=Pipeline([('scaler',StandardScaler()),
                       ('classification',model)])
    score=cross_val_score(pipeline,x,y,cv=5)
    print(f'Accuracy score: {np.mean(score)*100 :.2f}% of {model}')
