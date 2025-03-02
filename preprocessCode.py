import cv2
import os
import numpy as np

# Define paths
input_folder = 'NonDrowsyImages/'  # Folder with original images
output_folder = 'preprocessedNon_Drowsy_images/'  # Folder for preprocessed images

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define image size (128x128)
desired_size = (128, 128)
# Loop through all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check for image file types
        # Load the image
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Error loading image {filename}. Skipping.")
            continue

        # Step 1: Resize the image
        resized_image = cv2.resize(image, desired_size)

        # Step 2: Convert the image to grayscale (optional)
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Step 3: Normalize the pixel values to be between 0 and 1
        normalized_image = gray_image / 255.0

        # Step 4: Convert normalized image back to 0-255 scale for saving
        img_to_save = (normalized_image * 255).astype(np.uint8)

        # Step 5: Save the preprocessed image to the output folder
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, img_to_save)

        print(f"Preprocessed {filename} and saved to {save_path}")

print("Preprocessing completed.")
