# -*- coding: utf-8 -*-
# Simple Linear Regression on Head and Brain Size Dataset using Sklearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
file_path = "headbrain.csv"  # Change path if needed
data = pd.read_csv(file_path)

# Extract relevant columns
X = data["Head Size(cm^3)"].values.reshape(-1, 1)
y = data["Brain Weight(grams)"].values

# Train linear regression model
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)

# Plot data and regression line
plt.scatter(X, y, color='green', label='Data points')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('Head Size (cm³)')
plt.ylabel('Brain Weight (grams)')
plt.legend()
plt.show()

# Print model parameters and R² score
print(f"Sklearn Linear Regression: Slope = {reg.coef_[0]:.4f}, Intercept = {reg.intercept_:.4f}, R² = {reg.score(X, y):.4f}")
