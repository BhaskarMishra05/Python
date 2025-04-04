# You are given a dataset containing the advertising budget (in $1000s) spent on different platforms
# (TV, Radio, Social Media) and their corresponding sales (in $1000s). 
# Your task is to build a Multiple Linear Regression Model to predict
# sales based on the given advertising budgets.
import pandas as pd
import numpy as np
np.random.seed(42)
tv_budget=np.random.uniform(5,300,100)
radio_budget=np.random.uniform(1,100,100)
social_media_budget=np.random.uniform(1,200,100)
sales = 5 + (0.05 * tv_budget) + (0.07 * radio_budget) + (0.04 * social_media_budget) + np.random.normal(0, 2, 100)
df=pd.DataFrame({"TV":tv_budget,"radio":radio_budget,"social":social_media_budget,"sale":sales})

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
model=LinearRegression()
x=df[['TV','radio','social']]
y=df['sale']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
model.fit(x_train_scaled,y_train)
model_prediction=model.predict(x_test_scaled)
from sklearn.metrics import mean_squared_error,r2_score
f1=r2_score(y_test,model_prediction)
accuracy=mean_squared_error(y_test,model_prediction)
print(accuracy)
print(f1*100)
# Display actual vs. predicted values for the first 10 samples
print("\nActual vs. Predicted Sales:")
comparison_df = pd.DataFrame({"Actual Sales": y_test.values, "Predicted Sales": model_prediction})
print(comparison_df.head(10))
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(y_test, model_prediction, color="blue", alpha=0.6, label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Perfect Prediction")

plt.xlabel("Actual Sales ($1000s)")
plt.ylabel("Predicted Sales ($1000s)")
plt.title("Actual vs. Predicted Sales")
plt.legend()
plt.show()
