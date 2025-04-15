import numpy as np
import pandas as pd
np.random.seed(42)
n=500 
data={
'Brand':np.random.choice(['Honda', 'Toyota', 'BMW', 'Hyundai', 'Ford'],n),
'Model_Year' :np.random.randint(2000,2025,n),
'Mileage_kmpl':np.random.randint(20,70,n),
'Fuel_Type': np.random.choice(['Petrol', 'Diesel', 'CNG', 'Electric'],n),
'Transmission': np.random.choice(['Manual', 'Automatic'],n),	
'Engine_CC':np.random.uniform (1000 ,3000 ,n),
'Seating_Capacity':np.random.randint(4,8,n),
'Owner_Type':np.random.choice(['First', 'Second', 'Third'],n),
'Kilometers_Driven'	: np.random.uniform(5000,200000,n)}
brand_map={'Honda':50000, 'Toyota':45000, 'BMW':70000, 'Hyundai':35000, 'Ford':42000}
fuel_map={'Petrol':15000, 'Diesel':20000, 'CNG':25000, 'Electric':40000}
transmission_map={'Manual':47000, 'Automatic':46000}
owner_map={'First':75000, 'Second':55000, 'Third':30000}
df=pd.DataFrame(data)
base_price=150000
price_of_cars_predicted= (base_price+
                          df['Brand'].map(brand_map) + df['Mileage_kmpl'] *2.13 +
                          df['Fuel_Type'].map(fuel_map) +
                          df['Transmission'].map(transmission_map)+
                          df['Engine_CC'] * 1.45 +
                          df['Seating_Capacity']*1.89 +
                          df['Owner_Type'].map(owner_map) -
                          df['Kilometers_Driven']*1.05 -                      
                          df['Model_Year']*1.6)
df['Price']=np.round(price_of_cars_predicted)
x=df.drop('Price',axis=1)
y=df['Price']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
preprocessing=ColumnTransformer([('scaler',StandardScaler(),['Model_Year','Mileage_kmpl',
                                                             'Engine_CC','Seating_Capacity','Kilometers_Driven']),
                                 ('encoder',OneHotEncoder(drop='first'),['Brand','Fuel_Type','Transmission','Owner_Type'])])
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
model=Pipeline([('pre',preprocessing),
                ('linear',LinearRegression())])
model.fit(x_train,y_train)
prediction=model.predict(x_test)
from sklearn.metrics import r2_score,mean_squared_error
print(f'R2 Score: {r2_score(y_test,prediction) *100 :.2f}')
print(f'MSE: {mean_squared_error(y_test,prediction) :.2f}')
for actual,predicted in zip(y_test[:10],prediction[:10]):
    print(f'Actual: {actual :.2f}    | Predicted: {predicted  :.2f}'  )