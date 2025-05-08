from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
# Train decision tree
model = DecisionTreeClassifier()
model.fit(X, y)
y_pred = model.predict(X)
print(classification_report(y, y_pred))
 # Visualize the tree
plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree - Iris Dataset")
plt.show()