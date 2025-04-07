'''🧠 Problem Statement: Predict Music Genre Based on Audio Features
You're provided a synthetic dataset representing audio features of songs.
Your task is to classify the genre of a song based on its attributes using a classification model.
Then, show Actual vs Predicted genre labels in a readable format.

📦 Dataset Features:
tempo – Beats per minute

energy – Loudness & intensity

danceability – How suitable the track is for dancing

genre (🎯 Target):

Pop

Rock

Jazz

Classical

'''
import pandas as pd
import numpy as np
np.random.seed(42)
np_sample=100
tempo =np.random.uniform(1,60,np_sample)

energy =np.random.uniform(0.2,1.0,np_sample)

danceability = np.random.uniform(1,100,np_sample)

genre=np.random.choice(['Pop','Rock','Jazz','Classical'],np_sample)

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

df=pd.DataFrame({'tempo':tempo,'energy':energy,'dance':danceability,'genre':genre})

preprocessing=ColumnTransformer([('scaler',StandardScaler(),(['tempo','energy','dance']))])


x=df[['tempo','energy','dance']]
y=df['genre']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
model=Pipeline([('pre',preprocessing),
                ('knn',KNeighborsClassifier())])
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report
print(f'Accuracy score: {accuracy_score(y_test,y_pred)*100 :.2f}')
print(f'Classification report: {classification_report(y_test,y_pred)}')
for actual,predict in zip(y_test[:10],y_pred[:10]):
    print(f'Actual: {actual}    | Predict: {predict}')
