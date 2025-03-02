import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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
print(f"Training data size: {X_train.shape}")
print(f"Testing data size: {X_test.shape}")

# Step 4: Train Random Forest model
print("Training Random Forest...")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("Random Forest training complete.")

# Evaluate Random Forest
print("\nRandom Forest Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions))
print("Classification Report:\n", classification_report(y_test, rf_predictions))

# Step 5: Train SVM model
print("\nTraining Support Vector Machine (SVM)...")
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
print("SVM training complete.")

# Evaluate SVM
print("\nSVM Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, svm_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_predictions))
print("Classification Report:\n", classification_report(y_test, svm_predictions))

# Step 6: Hyperparameter tuning for Random Forest
print("\nHyperparameter tuning for Random Forest...")
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                              param_grid=rf_params, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)
rf_best_model = grid_search_rf.best_estimator_
print("Best parameters for Random Forest:", grid_search_rf.best_params_)

# Step 7: Hyperparameter tuning for SVM
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

# Step 8: Save the best models
print("\nSaving trained models...")
joblib.dump(rf_best_model, 'random_forest_model.pkl')
joblib.dump(svm_best_model, 'svm_model.pkl')
print("Models saved successfully.")
