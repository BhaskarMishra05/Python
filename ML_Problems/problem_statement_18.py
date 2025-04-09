'''🎨 Problem Statement: Predict Art Style from Image Features
You're given a synthetic dataset representing visual features of artworks.
Each artwork belongs to one of four styles: Impressionism, Cubism, Surrealism, or Abstract.

Your task is to:

Build a classification model to predict the art style based on the features.

Display the Actual vs Predicted art styles for the first few test samples.

📦 Dataset Features:
color_depth – Depth of color usage (scale: 0–100)

texture_complexity – How complex the texture is (scale: 1–10)

edge_sharpness – Sharpness of edges (scale: 1–5)

art_style (🎯 Target): Impressionism, Cubism, Surrealism, Abstract'''
import numpy as np
import pandas as pd
np.random.seed(42)
np_sample=100

color_depth =np.random.uniform(0,100,np_sample)

texture_complexity = np.random.uniform(0,10,np_sample)

edge_sharpness = np.random.uniform(1,5,np_sample)

art_style = np.random.choice(['Impressionism', 'Cubism', 'Surrealism', 'Abstract'],np_sample) 

df=pd.DataFrame({'color':color_depth,'texture':texture_complexity,'edge':edge_sharpness,'art':art_style})
x=df[['color','texture','edge']]
y=df['art']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
preprocessor=ColumnTransformer([('scaler',StandardScaler(),['color','edge','texture'])])
model=Pipeline([('preprocessor',preprocessor),
                ('classifier',KNeighborsClassifier())])
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score
print(f'Accuracy score : {accuracy_score(y_test,y_pred)*100 :.2f}')
for actual,predicted in zip(y_test[:10],y_pred[:10]):
    print(f'Acutal : {actual}   | Predicted : {predicted}')