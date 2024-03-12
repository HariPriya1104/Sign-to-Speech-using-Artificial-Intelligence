# Import necessary libraries
import os  # For interacting with the operating system
import cv2  # For accessing the webcam and image processing

# Define the data directory
DATA_DIR = 'data'

# Create the data directory if it does not exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the number of classes, starting index, and dataset size
number_of_classes = 10  # Total number of classes for data collection
start_from = 0  # Start index for the classes
dataset_size = 1000  # Number of images to capture per class

# Initialize the webcam feed
cap = cv2.VideoCapture(0)

# Loop through the classes
j = start_from
for j in range(start_from, number_of_classes):
    # Create directories for each class in the data directory
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    # Print a message indicating the current class being processed
    print('Collecting data for class {}'.format(j))

    done = False

    # Display message and wait for 'Q' key press to start capturing data
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(5) == ord('q'):
            break

    counter = 0
    # Continuously capture frames and save them as images in the respective class directories
    while counter < dataset_size:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow('frame', frame)
        cv2.waitKey(5)

        # Save the captured frame as an image in the appropriate class directory
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
