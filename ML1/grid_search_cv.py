import numpy as np
from sklearn import datasets 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Load dataset
iris = datasets.load_iris()
x = iris.data 
y = iris.target

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Define hyperparameters
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan']
}

# Create GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=5)

# Fit model
grid_search.fit(x, y)

# Output best results
print(f"Best Score: {grid_search.best_score_ * 100:.2f}%")
print("Best Parameters:", grid_search.best_params_)


