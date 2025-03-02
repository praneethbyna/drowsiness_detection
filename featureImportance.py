import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the saved Random Forest model
rf_model = joblib.load('random_forest_model.pkl')

# Assuming you have the feature names from your dataset
feature_names = ['left_EAR', 'right_EAR', 'avg_EAR', 'MAR', 'lip_distance', 'mouth_width_height_ratio', 
                 'head_tilt', 'head_nod', 'eyebrow_distance', 'eyebrow_asymmetry', 'facial_droop_asymmetry']  # Replace with actual feature names

# Get feature importance
feature_importances = rf_model.feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Print the feature importance table
print(importance_df)

# Plot feature importance
plt.figure(figsize=(12, 8))  # Increase figure size for better spacing
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance', fontsize=12)  # Adjust font size for clarity
plt.ylabel('Features', fontsize=12)
plt.title('Feature Importance in Random Forest', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()  # Automatically adjust padding to avoid clipping
plt.show()
