import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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

# Step 4: Train Random Forest model
print("Training Random Forest...")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("Random Forest training complete.")

# Step 5: Evaluate Random Forest
print("\nRandom Forest Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions))
print("Classification Report:\n", classification_report(y_test, rf_predictions))

# Step 6: Hyperparameter tuning for Random Forest (Optional)
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

# Step 7: Save the best model
print("\nSaving Random Forest model...")
joblib.dump(rf_best_model, 'random_forest_model.pkl')
print("Random Forest model saved successfully.")
