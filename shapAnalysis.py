import shap
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split

# Step 1: Load the Trained Model and Dataset
print("Loading model and dataset...")
rf_model = joblib.load('random_forest_model.pkl')  # Replace with your saved model path
file_path = 'combined_features.csv'  # Replace with your dataset path
df = pd.read_csv(file_path)

# Define features (X) and target (y)
X = df.drop(columns=['drowsiness_state', 'image_name'])
y = df['drowsiness_state']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Create SHAP Explainer
print("Creating SHAP explainer...")
explainer = shap.TreeExplainer(rf_model)

# Step 3: Calculate SHAP Values
print("Calculating SHAP values...")
shap_values = explainer.shap_values(X_test)

# Step 4: Generate Visualizations
print("Generating visualizations...")

# Global Importance: Summary Plot
print("Generating SHAP summary plot...")
shap.summary_plot(shap_values[1], X_test)  # Class "drowsy"

# Local Importance: Force Plot
print("Generating SHAP force plot for a single prediction...")
single_sample = X_test.iloc[0]
shap.force_plot(explainer.expected_value[1], shap_values[1][0], single_sample)

# Dependence Plot
print("Generating SHAP dependence plot for 'head_tilt'...")
shap.dependence_plot('head_tilt', shap_values[1], X_test)

# Save Outputs (Optional)
print("Saving SHAP outputs...")
importance_summary = shap.summary_plot(shap_values[1], X_test, show=False)
plt.savefig('shap_summary_plot.png')
plt.close()

# Save force plot as HTML (requires additional dependency: shap.force_plot supports saving HTML plots)
shap.save_html("shap_force_plot.html", shap.force_plot(explainer.expected_value[1], shap_values[1][0], single_sample))

print("SHAP analysis complete. Visualizations saved.")
