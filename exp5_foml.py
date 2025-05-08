import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv('BankNote_Authentication.csv')

# Split dataset into features and target
X, y = data.drop(columns='class'), data['class']

def train_and_evaluate(test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    for activation in ['relu', 'logistic', 'tanh', 'identity']:
        mlp = MLPClassifier(max_iter=500, activation=activation, random_state=42)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        print(f"\nActivation: {activation} | Test size: {test_size}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

# Run evaluations for different test sizes
train_and_evaluate(0.2)
train_and_evaluate(0.3)
