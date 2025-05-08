from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

# Load data, apply PCA, and split
lfw_people = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X_pca = PCA(150, whiten=True).fit_transform(lfw_people.data)
X_train, X_test, y_train, y_test = train_test_split(X_pca, lfw_people.target, test_size=0.25, random_state=42)

# Train model and predict
model = SVC(kernel='linear', class_weight='balanced').fit(X_train, y_train)
y_pred = model.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred, target_names=lfw_people.target_names))

# Display images with predictions
fig, axes = plt.subplots(3, 5, figsize=(12, 8))

# Get indices for the test set images
test_indices = np.unique(X_test, axis=0, return_index=True)[1]

for i, ax in enumerate(axes.flat):
    # Get the index for the image in the original data
    idx = test_indices[i]
    
    # Display the image
    ax.imshow(lfw_people.images[idx], cmap='gray')
    
    # Get predicted and true labels
    pred, true = lfw_people.target_names[y_pred[i]], lfw_people.target_names[y_test[i]]
    
    # Set title with green/red color depending on correct prediction
    ax.set_title(f"Pred: {pred}\nTrue: {true}", fontsize=8, color='green' if pred == true else 'red')
    ax.axis('off')

plt.tight_layout()
plt.show()
