#Problem Statement:
#📌 You are given a dataset containing a car’s engine size (liters) and 
#its corresponding fuel efficiency (miles per gallon - MPG).
#Your task is to build a Regression Model to predict a car’s MPG based on its engine size.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)
engine_size=np.random.uniform(1,6,100)
mpg = 50 - (engine_size * 5) + np.random.normal(0, 2, 100) 
df=pd.DataFrame({"Size of engine":engine_size,"MPG":mpg})
from sklearn.model_selection import train_test_split
x=df[["Size of engine"]]
y=df["MPG"]
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train_scaled,y_train)
model_prediction=model.predict(x_test_scaled)
from sklearn.metrics import r2_score,mean_squared_error
r2 = r2_score(y_test,model_prediction)
mse=mean_squared_error(y_test,model_prediction)
print(f"R2 sscore: {r2 *100 :.2f}")
print(f"MSE score :{mse :.2f} ")
for actual,prediction in zip(y_test[:10],model_prediction[:10]):
    print(f"Actual: L{actual:.2f}   |   Predicted: L{prediction:.2f}")

