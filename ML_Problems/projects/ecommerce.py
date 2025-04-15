import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
def data_entry(n=600,random_state=42):
    np.random.seed(random_state)
    data={ 'Product_Category':np.random.choice( ['Clothing', 'Electronics', 
                                        'Home Decor', 'Books', 'Beauty'],n),
    'Price': np.random.uniform(100,10000,n),
    'Discount_Percentage': np.random.randint(0,60,n),
    'Customer_Rating': np.random.uniform(1,5,n),
    'Shipping_Duration': np.random.randint(1,10,n),
    'Customer_Location': np.random.choice(['Urban', 'Semi-Urban', 'Rural'],n),
    'Payment_Mode': np.random.choice(['COD', 'Prepaid'],n),
    'Previous_Returns': np.random.randint(0,10,n),
    'Return': np.random.choice([0,1],n) } # (Target) 1 if returned, 0 otherwise
    df=pd.DataFrame(data)
    return df

# Class
class returnpredictur:
    def __init__(self):
        self.model =None
        self.preprocessing=ColumnTransformer([('scaler',StandardScaler(),['Price','Discount_Percentage',
                                                         'Customer_Rating','Shipping_Duration',
                                                         'Previous_Returns']),
                                              ('encoder',OneHotEncoder(drop='first'),
                                               ['Product_Category','Customer_Location','Payment_Mode'])])
        self.pipeline=Pipeline([('pre',self.preprocessing),
                                ('RFR',RandomForestClassifier())])
    
    def train(self,x,y):
        self.pipeline.fit(x,y)
    
    def predict(self,x):
        return self.pipeline.predict(x)
        
    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {acc*100:.2f}%')
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nSample Predictions:")
        for actual, pred in zip(y_test[:10], y_pred[:10]):
            print(f'Actual: {actual}  |  Predicted: {pred}')
    
    
# Call Function to call train_test_split
df= data_entry()
x=df.drop('Return',axis=1)
y=df['Return']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Call class to initiate predict
predictor=returnpredictur()
predictor.train(x_train,y_train)
predictor.evaluate(x_train,y_train)