import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

def generate_bike_dataset(n=600, random_state=42):
    np.random.seed(random_state)
    data = {
        'Brand': np.random.choice(['Hero', 'Honda', 'Yamaha', 'Royal Enfield', 'Bajaj', 'Suzuki'], n),
        'Model_Year': np.random.randint(2005, 2024, n),
        'Mileage_kmpl': np.random.randint(35, 90, n),
        'Engine_CC': np.random.randint(90, 750, n),
        'Fuel_Type': np.random.choice(['Petrol', 'Electric'], n),
        'Transmission': np.random.choice(['Manual', 'Automatic'], n),
        'Owner_Type': np.random.choice(['First', 'Second', 'Third'], n),
        'Kilometers_Driven': np.random.uniform(1000, 150000, n)
    }

    brand_map = {'Hero': 45000, 'Honda': 42000, 'Yamaha': 50000,
                 'Royal Enfield': 80000, 'Bajaj': 38000, 'Suzuki': 49000}
    fuel_map = {'Petrol': 10000, 'Electric': 20000}
    trans_map = {'Manual': 25000, 'Automatic': 25000}
    owner_map = {'First': 40000, 'Second': 20000, 'Third': 10000}

    df = pd.DataFrame(data)
    base_price = 20000

    price = (base_price +
             df['Brand'].map(brand_map) +
             df['Model_Year'] * 1.23 +
             df['Mileage_kmpl'] * 1.56 +
             df['Engine_CC'] * 2.21 +
             df['Fuel_Type'].map(fuel_map) +
             df['Transmission'].map(trans_map) +
             df['Owner_Type'].map(owner_map) -
             df['Kilometers_Driven'] +
             np.random.normal(1, 10, n))

    df['Price'] = np.round(price)
    return df
class BikePricePredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = ColumnTransformer([
            ('scaler', StandardScaler(), ['Model_Year', 'Mileage_kmpl', 'Engine_CC', 'Kilometers_Driven']),
            ('encoder', OneHotEncoder(drop='first'), ['Brand', 'Fuel_Type', 'Transmission', 'Owner_Type'])
        ])
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', LinearRegression())
        ])

    def train(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f'R² Score: {r2*100:.2f}')
        print(f'Mean Squared Error: {mse:.2f}')
        for actual, pred in zip(y_test[:10], y_pred[:10]):
            print(f'Actual: {actual:.2f}  |  Predicted: {pred:.2f}')
# Generate the data
df = generate_bike_dataset()
X = df.drop('Price', axis=1)
y = df['Price']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate model using the class
predictor = BikePricePredictor()
predictor.train(x_train, y_train)
predictor.evaluate(x_test, y_test)
