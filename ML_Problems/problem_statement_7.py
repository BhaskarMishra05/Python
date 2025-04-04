# You are given a dataset containing years of experience,
# education level, and job position of employees. 
# Your task is to predict their salaries using a Pipeline with 
# preprocessing steps and a Regression Model.
import numpy as np
import pandas as pd
np.random.seed(42)
num_samples=100
years_experience=np.random.uniform(1,20,num_samples)
education_levels = np.random.choice(["Bachelor", "Master", "PhD"], num_samples)
job_positions = np.random.choice(["Intern", "Junior", "Mid-Level", "Senior", "Manager"], num_samples)
base_salary = 30000
salary = (
    base_salary + 
    (years_experience * 2000) + 
    (np.where(education_levels == "Master", 5000, 0)) + 
    (np.where(education_levels == "PhD", 10000, 0)) +
    (np.where(job_positions == "Junior", 10000, 0)) +
    (np.where(job_positions == "Mid-Level", 20000, 0)) +
    (np.where(job_positions == "Senior", 40000, 0)) +
    (np.where(job_positions == "Manager", 60000, 0)) +
    np.random.normal(0, 5000, num_samples) 
)
df = pd.DataFrame({
    "Years Experience": years_experience,
    "Education Level": education_levels,
    "Job Position": job_positions,
    "Salary": salary
})
from sklearn.model_selection import train_test_split
x=df[["Years Experience","Education Level","Job Position"]]
y=df["Salary"]
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures,OneHotEncoder
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer([
    ("num_scaler", StandardScaler(), ["Years Experience"]),  # Scale numerical data
    ("cat_encoder", OneHotEncoder(drop="first"), ["Education Level", "Job Position"])  # Encode categorical data
])
model=Pipeline([(("preprocess", preprocessor))
                ,("poly_feature",PolynomialFeatures(degree=2))
                ,("regression",LinearRegression())])
model.fit(x_train,y_train)
model_prediction=model.predict(x_test)
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(y_test,model_prediction)
r2=r2_score(y_test,model_prediction)
print(f"MSE : {mse}")
print(f"R2 score : {r2*100 :.2f}")
print("Actual vs Predicted")
con_act_pred=pd.DataFrame({'Actual':y_test.values,'Prediction':model_prediction})
print(con_act_pred.head(10))