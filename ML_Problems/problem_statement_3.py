import numpy as np
import pandas as pd
np.random.seed(42)
house_size=np.random.randint(400,3000,100)
house_price=0.2*house_size+np.random.normal(0,20,100) # Adding Noise
df = pd.DataFrame({"House Size (sq ft)": house_size, "Price (in $1000s)": house_price})
# print(df.head())
from sklearn.model_selection import train_test_split
x=df[["House Size (sq ft)"]]
y=df["Price (in $1000s)"]
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.preprocessing import StandardScaler
scaled=StandardScaler()
x_train_scaled=scaled.fit_transform(x_train)
x_test_scaled=scaled.transform(x_test)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train_scaled,y_train)
model_prediction=model.predict(x_test_scaled)
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(y_test,model_prediction)
r2=r2_score(y_test,model_prediction)
print(f"MSE score: {mse :.2f}")
print(f"R2 score: {r2*100 :.2f}")
print(" Display actual vs. predicted prices for at least 10 houses")
for actual,prediction in zip( y_test[:10] , model_prediction[:10]):
     print(f"Actual: ${actual:.2f}K   |   Predicted: ${prediction:.2f}K")
    
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.scatter(y_test, model_prediction, color="blue", alpha=0.6, label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Perfect Prediction")
plt.xlabel("Actual Price ($1000s)")
plt.ylabel("Predicted Price ($1000s)")
plt.title("Actual vs. Predicted House Prices")
plt.legend()
plt.show()