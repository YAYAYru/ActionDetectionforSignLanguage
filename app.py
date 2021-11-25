from first_tv_module import mediapipe_detection, draw_styled_landmarks, mp_holistic, mp_drawing, extract_keypoints,actions

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

THRESHOLD = 0.8
PATH_VIDEO = "/Users/volley84/yayay/git/github/yayayru/slsru_ml_tag/data/video/sl_sentence_DianaB_DimaG/ss1_9_c5.mp4"

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.load_weights('action_day_sign_language.h5')


colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)        
    return output_frame


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# 1. New detection variables
sequence = []
sentence = []
#cap = cv2.VideoCapture(PATH_VIDEO)
# Set mediapipe model 
cap = cv2.VideoCapture(0)


def putText_center(frame, text, color, size):
        font = cv2.FONT_HERSHEY_PLAIN
        textsize = cv2.getTextSize(text, font, 1, 2)[0]

        # get coords based on boundary
        textX = int((frame.shape[1] - textsize[0]) / 2)
        textY = int((frame.shape[0] + textsize[1]) / 2)
        

        cv2.putText(frame, text, (textX, textY), font, size, color, 7)

def video2keypoints_mediapipe(frame, holistic):
    # Make detections
    image, results = mediapipe_detection(frame, holistic)
    print(results)
        
    # Draw landmarks
    draw_styled_landmarks(image, results)
        
    # 2. Prediction logic
    keypoints = extract_keypoints(results)
    return image, keypoints    

def visual(image, res, sentence):
    #3. Viz logic
    if res[np.argmax(res)] > THRESHOLD: 
        if len(sentence) > 0: 
            if actions[np.argmax(res)] != sentence[-1]:
                sentence.append(actions[np.argmax(res)])
        else:
            sentence.append(actions[np.argmax(res)])

    if len(sentence) > 5: 
        sentence = sentence[-5:]

    # Viz probabilities
    image = prob_viz(res, actions, image, colors)
    return image, sentence

WINDOW_SIZE = 30
T_SEC_BEGIN_RUN = time.time()
T_SEC_TO_START = 200
T_SEC_REC = 31
id_frame = 0
predict_word = "Not"
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
        #t_sec = T_SEC_TO_START - int(time.time() - T_SEC_BEGIN_RUN)  
        t_sec = T_SEC_TO_START - id_frame
        #print("t_sec", t_sec)

        if t_sec<0 and t_sec>-T_SEC_REC:
            frame, keypoints = video2keypoints_mediapipe(frame, holistic)

            sequence.insert(0,keypoints)
            sequence = sequence[:WINDOW_SIZE]
            #sequence.append(keypoints)
            #sequence = sequence[-30:]
            
            if len(sequence) == WINDOW_SIZE:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predict_word = actions[np.argmax(res)]    
                print("predict_word", predict_word)
                frame, sentence = visual(frame, res, sentence) 

            cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(frame, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        elif t_sec>=0:
            putText_center(frame, str(abs(t_sec)), (255, 255, 255),10)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else :            
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            putText_center(frame, predict_word, (0, 255, 0), 5)


        cv2.putText(frame, str(id_frame), (25, 700), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # Show to screen
        cv2.imshow('OpenCV Feed', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        id_frame = id_frame + 1
    cap.release()
    cv2.destroyAllWindows()


