'''Problem Statement: Predict Taxi Fare Amount
You are given a synthetic dataset that contains details of taxi trips. 
Your task is to predict the fare amount based on the trip's characteristics using a regression model with proper preprocessing.

📦 Dataset Features:
Distance (km) – Distance covered during the trip

Passenger Count – Number of people on the trip

Time of Day – Morning / Afternoon / Evening / Night

City – City where the trip occurred

'''
import numpy as np
import pandas as pd
np_sample=1000
np.random.seed(42)

Distance= np.random.uniform(1,10,np_sample)

Passengers= np.random.randint(1,5,np_sample)

Time= np.random.choice(['Morning', 'Afternoon','Evening','Night'],np_sample)

City=np.random.choice(['New York', 'San Francisco', 'Chicago', 'Boston', 'Seattle'],np_sample)

fare=(3+(Distance*2)+(Passengers*1.5)+
      (np.where(Time=='Morning',2,0))+
      (np.where(Time=='Afternoon',2,0))+
      (np.where(Time=='Evening',3,0))+
      (np.where(Time=='Night',4,0))+
      (np.where(City=='New York',10,0))+
      (np.where(City=='San Francisco',10,0))+
      (np.where(City=='Chicago',9,0))+
      (np.where(City=='Boston',8,0))+
      (np.where(City=='Seattle',7,0))+
      np.random.normal(0,2,np_sample)
      )
df=pd.DataFrame({'distance':Distance,'city':City,'passangers':Passengers,'Fare':fare,'time':Time})
x=df[['distance','city','passangers','time']]
y=df['Fare']

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
preprocesses=ColumnTransformer([('scaler',StandardScaler(),['distance','passangers'])
                                ,('cat_encoder',OneHotEncoder(drop='first'),['city','time'])])
model=Pipeline([('preprocesses',preprocesses),
                ('regression',LinearRegression())])

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=(42),test_size=0.2)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import mean_absolute_error,r2_score
mse=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f'MSE: {mse}')
print(f'R2: {r2}')
print("Actual vs Predicted")
cor_act_pred=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(cor_act_pred.head(10))
