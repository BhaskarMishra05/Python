''' You are given a dataset containing house-related features such as:

Size (in square feet)

Number of Bedrooms

Number of Bathrooms

Location (City Name)

Your task is to predict the monthly rent of houses using a Regression Model with necessary preprocessing steps. '''

import pandas as pd
import numpy as np
np.random.seed(42)
num_samples=100
size_sqft = np.random.randint(500, 3000, num_samples)
bedrooms = np.random.randint(1, 5, num_samples)
bathrooms = np.random.randint(1, 4, num_samples)
locations=np.random.choice(["New York", "San Francisco", "Los Angeles", "Chicago", "Houston"], num_samples)
base_rent = 500  
rent = (base_rent + 
        (size_sqft * 1.2) + 
        (bedrooms * 300) + 
        (bathrooms * 200) + 
        (np.where(locations == "New York", 1500, 0)) + 
        (np.where(locations == "San Francisco", 2000, 0)) + 
        (np.where(locations == "Los Angeles", 1200, 0)) +
        (np.where(locations == "Chicago", 800, 0)) +
        (np.where(locations == "Houston", 500, 0)) +
        np.random.normal(0, 300, num_samples))
df = pd.DataFrame({
    "Size (sq ft)": size_sqft,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Location": locations,
    "Rent ($)": rent
})
from sklearn.model_selection import train_test_split
x=df[['Size (sq ft)','Bedrooms','Bathrooms','Location']]
y=df['Rent ($)']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer  
preprocessor=ColumnTransformer([("scaler",StandardScaler(),['Size (sq ft)','Bedrooms','Bathrooms'])
                                ,('cat_encoder',OneHotEncoder(drop='first'),['Location'])])
model=Pipeline([('preprocessor',preprocessor)
                ,('regression',LinearRegression())])
model.fit(x_train,y_train)
model_prediction=model.predict(x_test)
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(y_test,model_prediction)
r2=r2_score(y_test,model_prediction)
print(f"R2: {r2 :.2f}")
print(f"MSE : {mse}")
print("Actual vs Predicted")
con_act_pred=pd.DataFrame({'Actual':y_test.values,'Predicted':model_prediction})
print(con_act_pred.head(10))
