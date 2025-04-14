import pandas as pd
import numpy as np
# Predict Student Exam Scores
# You are given a dataset containing hours studied and previous test scores of students.
# Your task is to predict their final exam scores using a Pipeline with multiple preprocessing steps.
np.random.seed(42)
hours_studied=np.random.uniform(1,10,100)
previous_scores=np.random.uniform(30,100,100)
final_scores = 10 + (5 * hours_studied) + (0.5 * previous_scores) + np.random.normal(0, 5, 100)  # Final scores
df=pd.DataFrame({"hours":hours_studied,"previous score":previous_scores,"final score":final_scores})
from sklearn.model_selection import train_test_split
x=df[['hours','previous score']]
y=df['final score']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
model=Pipeline([('scaler',StandardScaler())
                ,('poly_feature',PolynomialFeatures(degree=4))
                ,('regression',LinearRegression())])
model.fit(x_train,y_train)
model_prediction=model.predict(x_test)
from sklearn.metrics import r2_score,mean_squared_error
r2=r2_score(y_test,model_prediction)
mse=mean_squared_error(y_test,model_prediction)
print(f"R2: {r2 *100 :.2f}")
print(f"MSE: {mse}")
print("Actual vs prediction")
con_act_pred=pd.DataFrame({"Actual":y_test.values,"Predicted": model_prediction})
print(con_act_pred.head(10))