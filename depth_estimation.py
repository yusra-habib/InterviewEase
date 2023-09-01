#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv 
import mediapipe as mp 


def depth_estimation(img,face_landmarks):
    
    point_left = (int(face_landmarks.landmark[145].x * img.shape[1]),
                    int(face_landmarks.landmark[145].y * img.shape[0]))
    point_right = (int(face_landmarks.landmark[374].x * img.shape[1]),
                    int(face_landmarks.landmark[374].y * img.shape[0]))
    
    w = ((point_right[0] - point_left[0])**2 + (point_right[1] - point_left[1])**2)**0.5
    W = 6.3
    f = 650
    distance = (W * f) / w  
    
    return distance      

def process_depth_estimation(img,face_mesh,cv,mp):
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    distance=0
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        distance=depth_estimation(img,face_landmarks)
    return distance


# In[ ]:




