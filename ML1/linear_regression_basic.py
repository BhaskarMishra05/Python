import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

# Generate noisy linear data
noise = np.random.normal(1, 10, 100)
time = np.random.uniform(1, 50, 100)
score = 2 * time + noise
df = pd.DataFrame({'score': score, 'time': time})

# Mean squared error function
def mean_squared_error(m, b, points):
    total_errors = 0
    for i in range(len(points)):
        x = points.iloc[i].time 
        y = points.iloc[i].score
        total_errors += (y - (m * x + b)) ** 2
    return total_errors / float(len(points))

# Gradient descent
def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i].time 
        y = points.iloc[i].score
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

# Training
m = 0 
b = 0 
L = 0.0001
epochs = 1000

for i in range(epochs):
    m, b = gradient_descent(m, b, df, L)

print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")
print(f"MSE: {mean_squared_error(m, b, df)}")

# Plotting
plt.scatter(time, score, color='black', label='Data Points')
x_vals = np.linspace(min(time), max(time), 100)
y_vals = m * x_vals + b
plt.plot(x_vals, y_vals, color='red', label='Regression Line')
plt.xlabel("Time")
plt.ylabel("Score")
plt.title("Linear Regression via Gradient Descent")
plt.legend()
plt.show()
