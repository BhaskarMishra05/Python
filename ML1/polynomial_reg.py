import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)
#Housing dataset
house_size = np.random.randint(400, 3000, 100)
house_price = 0.05 * (house_size ** 1.5) + np.random.normal(0, 5000, 100)
df=pd.DataFrame({'size':house_size,'price':house_price})
from sklearn.model_selection import train_test_split
x=df[['size']]
y=df['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
# creating pipeline
degree=2
model=Pipeline([('poly_feature',PolynomialFeatures(degree=degree))
                ,('scaler',StandardScaler())
                ,('regressor',LinearRegression())])
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"MSE: {mse}")
print(f"R2: {r2*100 :.2f}")
plt.figure(figsize=(8, 5))
plt.scatter(x_test, y_test, color="blue", label="Actual Prices", alpha=0.6)
plt.scatter(x_test, y_pred, color="red", label="Predicted Prices", alpha=0.6)
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price ($1000s)")
plt.title(f"Polynomial Regression (Degree {degree})")
plt.legend()
plt.show()