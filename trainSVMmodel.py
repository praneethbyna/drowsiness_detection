import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Step 1: Load the dataset
file_path = 'combined_features.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Step 2: Define features (X) and target (y)
X = df.drop(columns=['drowsiness_state', 'image_name'])
y = df['drowsiness_state']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train SVM model
print("Training Support Vector Machine (SVM)...")
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
print("SVM training complete.")

# Step 5: Evaluate SVM
print("\nSVM Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, svm_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_predictions))
print("Classification Report:\n", classification_report(y_test, svm_predictions))

# Step 6: Hyperparameter tuning for SVM (Optional)
print("\nHyperparameter tuning for SVM...")
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

grid_search_svm = GridSearchCV(estimator=SVC(probability=True, random_state=42),
                               param_grid=svm_params, cv=5, scoring='accuracy')
grid_search_svm.fit(X_train, y_train)
svm_best_model = grid_search_svm.best_estimator_
print("Best parameters for SVM:", grid_search_svm.best_params_)

# Step 7: Save the best model
print("\nSaving SVM model...")
joblib.dump(svm_best_model, 'svm_model.pkl')
print("SVM model saved successfully.")
