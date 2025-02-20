import tensorflow as tf
from tensorflow import keras
import numpy as np

# Create sample data
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])  # XOR problem

# Define a simple neural network
model = keras.Sequential([
    keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=100, verbose=0)

# Test the model
predictions = model.predict(x_train)
print("Predictions:", predictions)
