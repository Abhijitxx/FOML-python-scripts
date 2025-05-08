import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

ada_boost = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_boost.fit(X_train, y_train)

y_pred_ada = ada_boost.predict(X_test)

print("AdaBoost Model Accuracy:", accuracy_score(y_test, y_pred_ada))
print("\nAdaBoost Classification Report:")
print(classification_report(y_test, y_pred_ada))

cm_ada = confusion_matrix(y_test, y_pred_ada)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_ada, annot=True, fmt='d', cmap='Blues', xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - AdaBoost on Digits Dataset')
plt.tight_layout()
plt.show()

grad_boost = GradientBoostingClassifier(n_estimators=50, random_state=42)
grad_boost.fit(X_train, y_train)

y_pred_grad = grad_boost.predict(X_test)

print("Gradient Boosting Model Accuracy:", accuracy_score(y_test, y_pred_grad))
print("\nGradient Boosting Classification Report:")
print(classification_report(y_test, y_pred_grad))

cm_grad = confusion_matrix(y_test, y_pred_grad)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_grad, annot=True, fmt='d', cmap='Blues', xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Gradient Boosting on Digits Dataset')
plt.tight_layout()
plt.show()
