import pandas as pd

# File paths
drowsy_file = 'features_phase1_drowsy.csv'        # Replace with the actual file path
non_drowsy_file = 'features_phase1_non_drowsy.csv'  # Replace with the actual file path

# Step 1: Load the data
drowsy_df = pd.read_csv(drowsy_file)
non_drowsy_df = pd.read_csv(non_drowsy_file)

# # Step 2: Add labels
# drowsy_df['drowsiness_state'] = 1  # Drowsy images
# non_drowsy_df['drowsiness_state'] = 0  # Non-drowsy images

# Step 3: Combine the data
combined_df = pd.concat([drowsy_df, non_drowsy_df], ignore_index=True)

# Step 4: Shuffle the dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 5: Save the combined dataset
combined_df.to_csv('combined_features.csv', index=False)

print("Combined dataset saved successfully!")
