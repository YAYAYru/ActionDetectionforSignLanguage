from first_tv_module import mediapipe_detection, draw_styled_landmarks, mp_holistic, mp_drawing, extract_keypoints, actions, model, DATA_PATH, no_sequences, sequence_length

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

