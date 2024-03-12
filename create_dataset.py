# Import necessary libraries
import os  # For interacting with the operating system
import pickle  # For serializing and de-serializing Python objects
import mediapipe as mp  # For utilizing the MediaPipe library for hand tracking
import cv2  # For image processing
import matplotlib.pyplot as plt  # For visualization purposes

# Set up MediaPipe for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8)

# Define the data directory

DATA_DIR = 'data/'

# Initialize lists to store data and corresponding labels
data = []
labels = []

# Loop through the directories in the data directory
for dir_ in os.listdir(DATA_DIR):
    # Loop through image files in the current directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        # Initialize lists for storing x and y coordinates
        x_ = []
        y_ = []

        # Read the image and convert it to RGB format
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to extract hand landmarks
        results = hands.process(img_rgb)

        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            # Loop through each detected hand
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract and store the x and y coordinates of each landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize the data and append it to the data_aux list
                data_aux = []
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Append the normalized data to the main data list and the label to the labels list
            data.append(data_aux)
            labels.append(dir_)
            print(dir_)

# Save the data and labels as a dictionary in a pickle file
f = open('Models\data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
