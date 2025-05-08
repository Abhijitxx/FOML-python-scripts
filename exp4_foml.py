import numpy as np
from sklearn.linear_model import Perceptron

X = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
y = np.array([0, 0, 1, 0])

model = Perceptron(max_iter=10000, eta0=0.1, random_state=0)
model.fit(X, y)

print("Final Weights:\n", model.coef_)
print("Final Bias:\n", model.intercept_)

def predict(inputs):
    return model.predict([inputs])

test_cases = [[1, 0], [1, 1], [0, 0], [0, 1]]
for test in test_cases:
    print(f"Input: {test} -> Prediction: {predict(test)}")
