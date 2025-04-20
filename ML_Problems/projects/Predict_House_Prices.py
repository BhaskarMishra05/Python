import numpy as np
import pandas as pd
np.random.seed(42)
n=500
data={'Area_sqft': np.random.uniform(400,4000,n),
'Bedrooms': np.random.uniform(1,6,n),
'Bathrooms': np.random.uniform(1,6,n),
'Floor': np.random.uniform(1,20,n),
'Total_Floors': np.random.uniform(1,30,n),
'Location':np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Pune'],n),
'Furnishing': np.random.choice(['Furnished', 'Semi-Furnished', 'Unfurnished'],n),
'Age_of_Property': np.random.randint(0,30,n)}

# Assign VAlues: 
location_map={'Mumbai':400000, 'Delhi':380000, 
              'Bangalore':450000, 'Hyderabad':350000, 'Chennai':320000, 'Pune':300000}
furnishing_map={'Furnished':500000, 'Semi-Furnished':30000, 'Unfurnished':50000}
df=pd.DataFrame(data)
base_price=400000
price=(base_price+df['Area_sqft'] * 900+
       df['Bedrooms'] *50000 + 
       df['Bathrooms'] * 20000 + df['Floor']*50000+
       df['Total_Floors']*100000 + 
       df['Location'].map(location_map)+
       df['Furnishing'].map(furnishing_map)-
       df['Age_of_Property'])
df["Price"]=np.round(price)

# Importing train_test_split
from sklearn.model_selection import train_test_split
x=df.drop('Price',axis=1)
y=df['Price']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

#Importing other modules
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
preprocessing=ColumnTransformer([('scaler',StandardScaler(),['Area_sqft','Bedrooms','Bathrooms',
                                                             'Total_Floors','Age_of_Property']),
                                 ('encoder',OneHotEncoder(drop='first'),['Location','Furnishing'])])

#Importing pipeline and model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
model=Pipeline([('pre',preprocessing),
                ('linear',LinearRegression())])
model.fit(x_train,y_train)
prediction=model.predict(x_test)

#Importing R2 and MSE
from sklearn.metrics import r2_score,mean_squared_error
print(f'R2 score: {r2_score(y_test,prediction) *100 :.2f}')
print(f'MSE: {mean_squared_error(y_test,prediction) :.2f}')
for actual,predicted in zip(y_test[:10],prediction[:10]):
       print(f'Actual: {actual :.2f}     | Predicted: {predicted :.2f}')