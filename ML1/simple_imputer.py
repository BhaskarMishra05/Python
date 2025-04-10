from sklearn.impute import SimpleImputer
import numpy as np
imputer=SimpleImputer(strategy='mean')
data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
data_impute=imputer.fit_transform(data)
print(data_impute)
# Takes avg and replace missing values with mean 
