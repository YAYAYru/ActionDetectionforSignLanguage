from first_tv_module import *
import cv2
import numpy as np
import random
import time
# BUG Когда закончится только видеофайл без нажатия на q, то есть появится какая-то ошибка
# File "D:\git\GitHub\YAYAYru\ActionDetectionforSignLanguage\code\first_tv_module.py", line 22, in mediapipe_detection
path_videofile = "../data/video/63_5_1.mp4" 
cap = cv2.VideoCapture(0) # camera
# cap = cv2.VideoCapture(path_videofile)

model = model_sequential()
model.load_weights(path_model)


# colors = [(245,117,16), (117,245,16), (16,0,245), (16,117,0), (0,117,245)]
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(actions.shape[0])]

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()    
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        #cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), color_rand, -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)        
    return output_frame

# 1. New detection variables
sequence = []
sentence = []
threshold = 0.8

WINDOW_SIZE = 30
T_SEC_BEGIN_RUN = time.time()
T_SEC_TO_START = 5

def putText_center(frame, text):
        font = cv2.FONT_HERSHEY_PLAIN
        textsize = cv2.getTextSize(text, font, 1, 2)[0]

        # get coords based on boundary
        textX = int((frame.shape[1] - textsize[0]) / 2)
        textY = int((frame.shape[0] + textsize[1]) / 2)
        
        cv2.putText(frame, text, (textX, textY), font, 10, (255, 255, 255), 10)



# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
        t_sec = T_SEC_TO_START - int(time.time() - T_SEC_BEGIN_RUN)  
        print("t_sec", t_sec)
        if t_sec<0 and t_sec>-10:
            # Make detections
            frame, results = mediapipe_detection(frame, holistic)
            print(results)
            
            # Draw landmarks
            draw_styled_landmarks(frame, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.insert(0,keypoints)
            sequence = sequence[:WINDOW_SIZE]
            #sequence.append(keypoints)
            #sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                
                
            #3. Viz logic
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                frame = prob_viz(res, actions, frame, colors)
                
            cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(frame, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        elif t_sec>=0:
            putText_center(frame, str(abs(t_sec)))
        else :
            putText_center(frame, "Predict")

        #if t_sec<0 and t_sec>-10:
        
        # Show to screen
        cv2.imshow('OpenCV Feed', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


