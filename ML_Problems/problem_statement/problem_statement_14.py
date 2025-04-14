'''Problem Statement: Predict Car Selling Price
You are working with a synthetic dataset of used cars.
Your task is to build a regression model to predict the selling price of a car based on its features.

📦 Dataset Features:
years— Year the car was manufactured
mileage — Number of kilometers the car has run
fuel_type — Type of fuel used (Petrol, Diesel, Electric, CNG)
transmission — Manual or Automatic
owner — Number of previous owners (0, 1, 2, 3)
selling_price — (🎯 Target) Selling price in usd
'''
import pandas as pd
import numpy as np
np.random.seed(42)
np_sample=100
years=np.random.uniform(1,15,np_sample)
milage=np.random.uniform(10,40,np_sample)
fule_type=np.random.choice(['Petrol','Diesel','Electric','CNG'],np_sample)
transmision=np.random.choice(['Manual','Automatic'],np_sample)
owners=np.random.uniform(0,3,np_sample)
base_price=30000
selling_price= (base_price-(years*1000)-(milage*1.3)+
                (np.where(fule_type=='Petrol',1200,0))+
                (np.where(fule_type=='Diesel',2000,0))+                
                (np.where(fule_type=='Electric',2500,0))+
                (np.where(fule_type=='CNG',1800,0))+
                (np.where(transmision=='Manual',2400,0))+
                (np.where(transmision=='Autometic',3000,0))-
                (owners*1500)+np.random.normal(1,10,np_sample))
df=pd.DataFrame({'years':years,'fule':fule_type,'transmission':transmision,'milage':milage,
                 'owners':owners,'selling price':selling_price})
x=df[['years','fule','transmission','owners','milage']]
y=df['selling price']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
preprossesing=ColumnTransformer([('scaler',StandardScaler(),['years','milage','owners']),
                                 ('encode',OneHotEncoder(drop='first'),['transmission','fule'])])
model=Pipeline([('preprossesing',preprossesing),
                ('linear',LinearRegression())])
model.fit(x_train,y_train)
model_prediction=model.predict(x_test)
from sklearn.metrics import mean_squared_error,r2_score
print(f'MSE: {mean_squared_error(y_test,model_prediction):.2f}')
print(f'R2: {r2_score(y_test,model_prediction)*100 :.2f}')
print("Actual vs Predicted")
cor_act_pred=pd.DataFrame({'Actual':y_test,'Predicted':model_prediction})
print(cor_act_pred.head(10))
import matplotlib.pyplot as plt
plt.scatter(y_test,model_prediction,color='black')
plt.show()
