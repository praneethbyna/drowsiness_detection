import cv2
import dlib
import os
import numpy as np
import pandas as pd



def extract_facial_droop_asymmetry(landmarks_points):
    # Asymmetry in the face (e.g., comparing the height difference between left and right sides of the face)
    # Compare the corners of the mouth (48, 54) or other facial features for droop
    left_mouth_corner = landmarks_points[48]
    right_mouth_corner = landmarks_points[54]

    # Calculate vertical asymmetry in the mouth
    facial_droop_asymmetry = abs(left_mouth_corner[1] - right_mouth_corner[1])

    return facial_droop_asymmetry


def extract_eyebrow_features(landmarks_points):
    # Eyebrow points: Left eyebrow (17-21), Right eyebrow (22-26)
    left_eyebrow = landmarks_points[17:22]
    right_eyebrow = landmarks_points[22:27]

    # Calculate the average distance between eyebrow points and the corresponding eye landmarks
    left_eyebrow_distance = np.mean([np.linalg.norm(landmarks_points[19] - landmarks_points[37]),  # Between eye and eyebrow
                                     np.linalg.norm(landmarks_points[21] - landmarks_points[38])])

    right_eyebrow_distance = np.mean([np.linalg.norm(landmarks_points[22] - landmarks_points[43]),
                                      np.linalg.norm(landmarks_points[24] - landmarks_points[44])])

    # Eyebrow asymmetry
    eyebrow_asymmetry = abs(left_eyebrow_distance - right_eyebrow_distance)

    return left_eyebrow_distance, eyebrow_asymmetry


def calculate_MAR(mouth):
    # Vertical distance
    A = np.linalg.norm(mouth[13] - mouth[19])  # Lip distance (top to bottom)
    B = np.linalg.norm(mouth[15] - mouth[17])

    # Horizontal distance
    C = np.linalg.norm(mouth[12] - mouth[16])

    # MAR formula
    MAR = (A + B) / (2.0 * C)
    return MAR

def extract_mouth_features(landmarks_points):
    # Mouth landmarks (48-67)
    mouth_points = landmarks_points[48:68]

    # Calculate MAR
    MAR = calculate_MAR(mouth_points)

    # Calculate other mouth features like lip distance and width-to-height ratio
    lip_distance = np.linalg.norm(mouth_points[13] - mouth_points[19])  # Top to bottom lip distance
    mouth_width_height_ratio = np.linalg.norm(mouth_points[12] - mouth_points[16]) / lip_distance

    return MAR, lip_distance, mouth_width_height_ratio


def calculate_EAR(eye):
    # Vertical distance
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # Horizontal distance
    C = np.linalg.norm(eye[0] - eye[3])

    # EAR formula
    EAR = (A + B) / (2.0 * C)
    return EAR

def extract_eye_features(landmarks_points):
    # Dlib eye landmark points: 
    # Left eye (36-41), Right eye (42-47)
    left_eye_points = landmarks_points[36:42]
    right_eye_points = landmarks_points[42:48]

    # Calculate EAR for both eyes
    left_EAR = calculate_EAR(left_eye_points)
    right_EAR = calculate_EAR(right_eye_points)

    # Average EAR
    avg_EAR = (left_EAR + right_EAR) / 2.0

    return left_EAR, right_EAR, avg_EAR


def extract_head_pose(landmarks_points):
    # Head pose calculation using key landmarks
    # Nose tip (30), Chin (8), Left eye corner (36), Right eye corner (45), 
    # Left mouth corner (48), Right mouth corner (54)
    
    nose_tip = landmarks_points[30]
    chin = landmarks_points[8]
    left_eye_corner = landmarks_points[36]
    right_eye_corner = landmarks_points[45]
    left_mouth_corner = landmarks_points[48]
    right_mouth_corner = landmarks_points[54]

    # Calculate head tilt as the horizontal angle of the line between the two eye corners
    eye_line = np.linalg.norm(right_eye_corner - left_eye_corner)
    tilt_angle = np.arctan2(right_eye_corner[1] - left_eye_corner[1], right_eye_corner[0] - left_eye_corner[0])
    
    # Calculate head nod as the vertical angle between the chin and nose tip
    nod_angle = np.arctan2(chin[1] - nose_tip[1], chin[0] - nose_tip[0])

    return tilt_angle, nod_angle



# Initialize Dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Download from Dlib's repository

# Prepare to store features
features_phase1 = []

# Define drowsiness state (1 for drowsy images)
non_drowsiness_state = 0

# Loop through images in the folder
image_folder = 'preprocessedNon_Drowsy_images'  # Folder containing drowsy images
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_folder, filename)
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks_points = np.array([(p.x, p.y) for p in landmarks.parts()])

            # Extract features (e.g., EAR, MAR, head tilt, etc.)
            left_EAR, right_EAR, avg_EAR = extract_eye_features(landmarks_points)
            MAR, lip_distance, mouth_width_height_ratio = extract_mouth_features(landmarks_points)
            head_tilt, head_nod = extract_head_pose(landmarks_points)
            eyebrow_distance, eyebrow_asymmetry = extract_eyebrow_features(landmarks_points)
            facial_droop_asymmetry = extract_facial_droop_asymmetry(landmarks_points)

            # Store features along with the image name and drowsiness state
            feature_data = {
                "image_name": filename,
                "drowsiness_state": non_drowsiness_state,  # Label as drowsy
                "left_EAR": left_EAR,
                "right_EAR": right_EAR,
                "avg_EAR": avg_EAR,
                "MAR": MAR,
                "lip_distance": lip_distance,
                "mouth_width_height_ratio": mouth_width_height_ratio,
                "head_tilt": head_tilt,
                "head_nod": head_nod,
                "eyebrow_distance": eyebrow_distance,
                "eyebrow_asymmetry": eyebrow_asymmetry,
                "facial_droop_asymmetry": facial_droop_asymmetry
            }

            # Append feature data to the list
            features_phase1.append(feature_data)
            print(filename+"done")

# Convert to DataFrame and save with image_name and drowsiness state as columns
df_phase1 = pd.DataFrame(features_phase1)
df_phase1.to_csv('features_phase1_non_drowsy.csv', index=False)
