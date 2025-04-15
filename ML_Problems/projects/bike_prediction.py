import numpy as np
import pandas as pd
n=600 
np.random.seed(42)
data={'Brand':np.random.choice(['Hero', 'Honda', 'Yamaha', 'Royal Enfield', 'Bajaj', 'Suzuki'],n),
'Model_Year' : np.random.randint(2005,2024,n),
'Mileage_kmpl'	: np.random.randint(35,90,n),
'Engine_CC':np.random.randint(90,750,n),
'Fuel_Type':np.random.choice(['Petrol', 'Electric'],n),
'Transmission' :np.random.choice(['Manual', 'Automatic'],n),
'Owner_Type':np.random.choice(['First', 'Second', 'Third'],n),	
'Kilometers_Driven':np.random.uniform(1000,150000,n)}

brand_map={'Hero':45000, 'Honda':42000, 'Yamaha':50000,
           'Royal Enfield':80000, 'Bajaj':38000, 'Suzuki':49000}
fuel_map={'Petrol':10000,'Electric':20000}
Transmission_map={'Manual':25000,'Automatic':25000}
owner_map={'First':40000,'Second':20000,'Third':10000}
df=pd.DataFrame(data)
base_price=20000
price_of_bike= (base_price+
                df['Brand'].map(brand_map)+
                df['Model_Year']*1.23 +
                df['Mileage_kmpl']*1.56 +
                df['Engine_CC']*2.21 +
                df['Fuel_Type'].map(fuel_map)+
                df['Transmission'].map(Transmission_map)+
                df['Owner_Type'].map(owner_map)-
                df['Kilometers_Driven']+
                np.random.normal(1,10,n))
df['Price']=np.round(price_of_bike)
from sklearn.model_selection import train_test_split
x=df.drop('Price',axis=1)
y=df['Price']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)


# IMPORTING STANDARD SCALER , ONEHOTENCODER, COLUMNTRANSFORMER, LINEAR REGRESSION MODEL WITH PIPELINE
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
preprocessing=ColumnTransformer([('scaler',StandardScaler(),['Model_Year','Mileage_kmpl',
                                                             'Engine_CC','Kilometers_Driven']),
                                 ('encoder',OneHotEncoder(drop='first'),['Brand','Fuel_Type',
                                                                         'Transmission','Owner_Type'])])
model=Pipeline([('pre',preprocessing),
               ('linear',LinearRegression())])
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#Import r2 and mse
from sklearn.metrics import r2_score,mean_squared_error
print(f'R2 Score: {r2_score(y_test,y_pred) *100 :.2f}')
print(f'MSE: {mean_squared_error(y_test,y_pred) :.2f}')
for actual ,predict in zip(y_test[:10],y_pred[:10]):
    print(f'Actual: {actual :.2f}     | Predicted: {predict  :.2f}')
