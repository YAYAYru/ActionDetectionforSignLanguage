from first_tv_module import *
import cv2
import numpy as np
import random
# BUG Когда закончится только видеофайл без нажатия на q, то есть появится какая-то ошибка
# File "D:\git\GitHub\YAYAYru\ActionDetectionforSignLanguage\code\first_tv_module.py", line 22, in mediapipe_detection
#path_videofile = "../data/video/63_5_1.mp4" 
path_videofile = "/Users/volley84/yayay/git/github/yayayru/slsru_ml_tag/data/video/sl_sentence_DianaB_DimaG/ss1_9_c5_1.mp4" 
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

# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.insert(0,keypoints)
        sequence = sequence[:30]
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
            image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


