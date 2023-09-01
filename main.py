

def libraries():
    import cv2 as cv 
    import numpy as np
    import mediapipe as mp 
    import math as math
    import time as time
    import requests
    import statistics as statistics 
    return cv,np,mp,math,time,statistics

#def saveImage(cv,frame):   
#    cv.imwrite("thirty_frame.jpg", frame) 
#    print("thirtith frame captured and saved.")
#    # Load the captured and saved image
#    saved_image_path = "thirty_frame.jpg"
#    frame = cv.imread(saved_image_path)

#    return frame

def scores_calc(total_Video_time,eyeTracking_values_after_3seconds,eyeStatus_values_after_3seconds,number_faces_after_3seconds,expression_values_after_3seconds): 
    scores = {'open': 3, 'closed': 1, 'one': 2, 'two': 0, 'zero': 0, 'closed lips': 1, 'smile': 3, 'neutral': 2, 'center': 3, 'top': 1, 'right': 1, 'left': 1, 'bottom': 1, 'None': 0}
    total_sum = 0
    
    for i in range(len(eyeTracking_values_after_3seconds)):
        for j in range(len(eyeTracking_values_after_3seconds[i])):
            value = scores.get(eyeTracking_values_after_3seconds[i][j], 0)
            eyeTracking_values_after_3seconds[i][j] = value
            total_sum += value
    for i in range(len(eyeStatus_values_after_3seconds)):
        value = scores.get(eyeStatus_values_after_3seconds[i], 0)
        eyeStatus_values_after_3seconds[i] = value
        total_sum += value
    for i in range(len(expression_values_after_3seconds)):
        value = scores.get(expression_values_after_3seconds[i], 0)
        expression_values_after_3seconds[i] = value
        total_sum += value

    for i in range(len(number_faces_after_3seconds)):
        value = scores.get(number_faces_after_3seconds[i], 0)
        number_faces_after_3seconds[i] = value
        total_sum += value
    return total_sum

def main():
    from eye_tracking import iris_position,eyes_closed,euclideanDistance,main_eye_tracking
    from expression_analysis import process_frame,smile,smileImage,euclideanDistance,process_frameImage,main_expression_analysis
    from face_count import main_face_count
    from depth_estimation import process_depth_estimation,depth_estimation
    #from eyebrows_analysis import *
    
    cv,np,mp,math,time,statistics=libraries()

    

    #initialize variable
    eyeStatus_values = []
    eyeTracking_valuesRight = []
    eyeTracking_valuesLeft = []
    expression_values = []
    number_faces=[]
    
    eyeStatus_values_after_3seconds = []
    eyeTracking_values_after_3seconds = []   
    expression_values_after_3seconds = []
    number_faces_after_3seconds=[]
    
    saveImag_counter=0
    cap = cv.VideoCapture(0)
    frame_rate = cap.get(cv.CAP_PROP_FPS)
    update_interval = 5#seconds
    start_time = time.time()
    

    
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            frame = cv.flip(frame, 1)
            if not ret:
                break
            
            saveImag_counter+=1
            if saveImag_counter==30:
                print('ready!!')
                #frame1=saveImage(cv,frame)
                frame1=frame
                distance=process_depth_estimation(frame1,face_mesh,cv,mp)
                frame1,mask1,total_dist_horizontal,total_dist_vertical = process_frameImage(frame1, face_mesh,cv,np,mp,math,time)
                frame_counter=0
                start_Video_time = time.time()
                
            if saveImag_counter>30:
                
                number_face=main_face_count(frame,cv,1)
                if number_face=='one':
                    frame,expression_value=main_expression_analysis(total_dist_horizontal,total_dist_vertical,distance,cv,np,mp,math,time,face_mesh,frame)
                    frame,eye_status,iris_pos_right,iris_pos_left= main_eye_tracking(frame, face_mesh,cv,np,mp,math,time,statistics)

                    eyeStatus_values.append(eye_status)
                    eyeTracking_valuesLeft.append(iris_pos_left)
                    eyeTracking_valuesRight.append(iris_pos_right)
                    number_faces.append(number_face)
                    expression_values.append(expression_value)



                    elapsed_time = time.time() - start_time
                    if elapsed_time >= update_interval:
                        most_common_expression_values = statistics.mode(expression_values)
                        most_common_eyeStatus = statistics.mode(eyeStatus_values)
                        most_common_facesNumber=statistics.mode(number_faces)
                        most_common_expression_left = statistics.mode(eyeTracking_valuesLeft)
                        most_common_expression_right = statistics.mode(eyeTracking_valuesRight)

                        eyeTracking_values_after_3seconds.append([most_common_expression_left,most_common_expression_right])
                        eyeStatus_values_after_3seconds.append(most_common_eyeStatus)
                        number_faces_after_3seconds.append(most_common_facesNumber)
                        expression_values_after_3seconds.append(most_common_expression_values)


                        start_time = time.time()
                        eyeStatus_values=[]
                        eyeTracking_valuesLeft=[]
                        eyeTracking_valuesRight=[]
                        expression_values=[]
                        number_faces=[]
                    
            cv.imshow('img', frame)
            key = cv.waitKey(1)
            if key ==ord('q'):
                break

                
     
    cap.release()
    cv.destroyAllWindows()
    current_Video_time = time.time()
    total_Video_time=current_Video_time-start_Video_time
    total_Video_time = math.floor(total_Video_time/4)
    
    total_score=total_Video_time*5
    if total_Video_time>0:
        score=scores_calc(total_Video_time,eyeTracking_values_after_3seconds,eyeStatus_values_after_3seconds,number_faces_after_3seconds,expression_values_after_3seconds)
        confidentScore=score/total_score
        #highest score 120/40=3 
        if confidentScore>2.5 and confidentScore<=3:
            print(f"Confidence level is high and confidence score is {confidentScore}")
        elif confidentScore>=1.5 and confidentScore<=2.5:
            print(f"Confidence level is medium and confidence score is {confidentScore}")
        elif confidentScore>=0 and confidentScore<1.5:
            print(f"Confidence level is low and confidence score is {confidentScore}")
        else:
            print('cannot calculate confidence score')
            
    else:
        print('some problem occur')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:








# In[ ]:





# In[ ]:





# In[ ]:




