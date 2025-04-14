# Problem Statement: Predict Car Fuel Efficiency (MPG)
# You're given a dataset containing various features about cars.
# Your task is to predict the miles per gallon (MPG) —
# a measure of fuel efficiency — using a regression model with necessary preprocessing.
import numpy as np
import pandas as pd
np_sample=100

Horsepower=np.random.uniform(90,200,np_sample)

Weight=np.random.uniform(950,2500,np_sample)

Cylinders=np.random.choice(['3','4','6','8','10','12'],np_sample)

Transmission=np.random.choice(['Manual','Automatic'],np_sample)

Transmission_encoded = np.where(Transmission == 'Manual', 1, 0)

Cylinders_numeric = Cylinders.astype(int)

fuel_efficiency = (
    50
    - 0.03 * Horsepower
    - 0.004 * Weight
    + 2 * Transmission_encoded
    - 1.5 * Cylinders_numeric
)

df=pd.DataFrame({'horsepower':Horsepower,'weight(kg)':Weight,'cylinders':Cylinders,'transmission':Transmission,'mpg':fuel_efficiency})
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
preprosses=ColumnTransformer([('scaler',StandardScaler(),['horsepower','weight(kg)']),
                              ('cat_encoder',OneHotEncoder(drop='first'),['cylinders','transmission'])])
x=df[['horsepower','weight(kg)','cylinders','transmission']]
y=df['mpg']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
model=Pipeline([('preprocessos',preprosses),
                 ('regression',LinearRegression())])
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"MSE: {mse}")
print(f"R2: {r2}")
print("Actual vs Predicted")
cor_act_pred=pd.DataFrame({'Actual ':y_test,'Predicted ':y_pred})
print(cor_act_pred.head(10))

import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Actual vs Predicted MPG")
plt.grid(True)
plt.show()
