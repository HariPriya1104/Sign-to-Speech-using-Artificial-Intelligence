import pickle
import cv2
import mediapipe as mp
import numpy as np
from Models.audio import say
import time
import threading
from queue import Queue

model_dict = pickle.load(open('Models\model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8)

# Dictionary to map numeric predictions to corresponding labels
labels_dict = {0: 'Hello', 1: 'I\'m hungry', 2: 'I need Water', 3: 'How are you', 4: 'I\'m Sleepy'}  # Mapping of numeric predictions to labels
margin = 20  # Margin for the bounding box around the detected hand

# Initialize previous prediction variable and timer
previous_prediction = None
last_reset_time = time.time()

# Queue to manage audio tasks
audio_queue = Queue()

# Function to say the predicted sign
def say_predicted_sign(predicted_sign):
    say(str(predicted_sign))

# Worker function to process audio tasks
def audio_worker():
    while True:
        predicted_sign = audio_queue.get()
        say_predicted_sign(predicted_sign)
        audio_queue.task_done()

# Start audio worker thread
audio_thread = threading.Thread(target=audio_worker)
audio_thread.daemon = True
audio_thread.start()

while True:
    try:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        frame=cv2.flip(frame,1)

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            if len(data_aux) != 42:
                print("Error: Unexpected number of features. Skipping prediction.")
                continue

            prediction = model.predict([np.asarray(data_aux)])

            predicted_sign = labels_dict[int(prediction[0])]
            
            cv2.rectangle(frame, (x1-margin, y1-margin), (x2+margin, y2+margin), (0, 0, 0), 4)
            cv2.putText(frame, predicted_sign, (x1, y1 - margin-10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
            
            # If the predicted sign is different from the previous one, add it to the audio queue
            if predicted_sign != previous_prediction:
                audio_queue.put(predicted_sign)
                previous_prediction = predicted_sign
            
            # Check if 5 seconds have elapsed since the last reset
            if time.time() - last_reset_time >=10:
                previous_prediction = None
                last_reset_time = time.time()

        cv2.imshow('frame', frame)
        k=cv2.waitKey(1)
        if(k==27):
            break
    except ValueError as e:
        print(f"ValueError: {e}. Skipping prediction.")
        cv2.imshow('frame', frame)
        continue

cap.release()
cv2.destroyAllWindows()
