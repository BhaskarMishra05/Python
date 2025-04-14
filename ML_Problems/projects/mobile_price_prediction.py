import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,mean_squared_error
np.random.seed(42)
n=500 
data={'RAM_GB': np.random.choice([2,4,6,8,12],n),
      'Storage_GB':np.random.choice([32,64,128,256,512,1024],n),
      'Battery_mAh':np.random.randint(3000,7000,n),
      'RearCam_MP':np.random.randint(32,108,n),
      'FrontCam_MP':np.random.randint(16,50,n),
      'Screen_inch':np.random.uniform(5,7,n),
      '5G_Support':np.random.choice(['Yes','No'],n),
      'Brand_Code':np.random.choice(['Samsung','Apple','Realme','Nokia','Xiaomi','Oppo','Vivo'])}

brand_map = {'Samsung': 6000, 'Apple': 10000, 'Realme': 4000, 'Nokia': 3000,
             'Xiaomi': 5000, 'Oppo': 4500, 'Vivo': 4000}
support_map = {'Yes': 5000, 'No': 0}

df=pd.DataFrame(data)

price = (
    df["RAM_GB"] * 1200 +
    df["Storage_GB"] * 20 +
    df["Battery_mAh"] * 0.5 +
    df["RearCam_MP"] * 50 +
    df["FrontCam_MP"] * 30 +
    df["Screen_inch"] * 1000 +
    df["5G_Support"].map(support_map) +
    df["Brand_Code"].map(brand_map) +
    np.random.normal(0, 5000, size=n)
)
# Add Price into dataset
df["Price_INR"] = np.round(price)

#Import train_test_split
x=df.drop('Price_INR',axis=1)
y=df['Price_INR']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

# Importing Column Transformer and creating Pipeline
preprocessing=ColumnTransformer([('scaler',StandardScaler(),['RAM_GB','Storage_GB','Battery_mAh',
                                                             'RearCam_MP','FrontCam_MP','Screen_inch']),
                                 ('encoder',OneHotEncoder(drop='first'),['5G_Support','Brand_Code'])])
model=Pipeline([('pre',preprocessing),
                ('linear',LinearRegression())])

# Fit and predict 
model.fit(x_train,y_train)
prediction=model.predict(x_test)

# Checking prediction
print(f'R2 score: {r2_score(y_test,prediction) *100 :.2f}')
print(f'MSE: {mean_squared_error(y_test,prediction) :.2f}')

for actual,predicted in zip(y_test[:10],prediction[:10]):
    print(f'Actual: {actual :.2f}      | Prediction: {predicted :.2f}')

