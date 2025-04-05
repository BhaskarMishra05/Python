#Problem Statement: Predict Student Monthly Study Hours
#You are given a dataset containing:
# Grade Level (e.g., 6th, 7th, 8th, etc.)
# Participation in Extracurricular Activities (Yes/No)
import pandas as pd
import numpy as np
np.random.seed(42)
np_limit=100
grade_levels=np.random.choice(['6th','7th','8th','9th','10th','11th','12th'],np_limit)
extracurricular_activity=np.random.choice(['Yes','No'],np_limit)
ages=np.random.randint(11,18,np_limit)
base_hours=20
study_hours = (
    base_hours +
    (ages * 1.5) +
    (np.where(extracurricular_activity == "Yes", -10, 5)) +  
    (np.array([int(g[:-2]) for g in grade_levels]) * 2) +
    np.random.normal(0, 5, np_limit) 
)
from sklearn.preprocessing import StandardScaler,PolynomialFeatures,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
df=pd.DataFrame({'grade':grade_levels,'extracurricular':extracurricular_activity,'ages':ages,'study hours':study_hours})
x=df[['grade','extracurricular','ages']]
y=df['study hours']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
preprocessor=ColumnTransformer([('scaler',StandardScaler(),['ages']),
                                ('cat_encode',OneHotEncoder(drop='first'),['grade','extracurricular'])])
model=Pipeline([('preprocess',preprocessor),
                ('regressor',LinearRegression())])
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"MSE: {mse}")
print(f"R2 score: {r2}")
print("Actual vs Predicted")
cor_act_pred=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(cor_act_pred.head(10))