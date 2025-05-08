from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# Generate a synthetic binary classification dataset with 1 informative feature
X, y = make_classification(
    n_samples=100,
    n_features=1,
    n_informative=1,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=0
)

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predict on the same dataset
y_pred = model.predict(X)

# Print evaluation metrics
print(classification_report(y, y_pred))

# Plot the original data points
plt.scatter(X, y, c=y, cmap='bwr', edgecolors='k')

# Create evenly spaced values across the feature range
x_vals = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

# Predict probability for each value
probs = model.predict_proba(x_vals)[:, 1]

# Plot the sigmoid curve (probability output)
plt.plot(x_vals, probs, color='black')

# Add plot labels and title
plt.title('Logistic Regression')
plt.xlabel('Feature')
plt.ylabel('Probability')
plt.show()
