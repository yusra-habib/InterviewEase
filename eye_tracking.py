#!/usr/bin/env python
# coding: utf-8

# In[1]:


#contain code 
#check if eyes closed or not 
#if closed then dont check for positions of eyes


def euclideanDistance(point1,point2,math):
    x1,y1=point1.ravel()
    x2,y2=point2.ravel()
    distance=math.sqrt(((x2-x1)**2)+((y2-y1)**2))
    return distance

def eyes_closed(L_H_RIGHT,L_H_LEFT,L_H_TOP,L_H_BOTTOM,R_H_RIGHT,R_H_LEFT,R_H_TOP,R_H_BOTTOM,math):
    eyes_closed="" 
    Right_horiz_dist=euclideanDistance(L_H_RIGHT,L_H_LEFT,math)
    Left_horiz_dist=euclideanDistance(L_H_RIGHT,L_H_LEFT,math)
    
    Right_ver_dist=euclideanDistance(L_H_TOP,L_H_BOTTOM,math)
    Left_ver_dist=euclideanDistance(R_H_TOP,R_H_BOTTOM,math)
    
    left_eye_ratio=6
    right_eye_ratio=6
    
    if Right_ver_dist<=0 and Left_ver_dist>0:
        left_eye_ratio=Left_horiz_dist/Left_ver_dist
        
    elif Right_ver_dist>0 and Left_ver_dist<=0:
        right_eye_ratio=Right_horiz_dist/Right_ver_dist
        
    else :
        right_eye_ratio=Right_horiz_dist/Right_ver_dist
        left_eye_ratio=Left_horiz_dist/Left_ver_dist
    
    #ratio=(right_eye_ratio+left_eye_ratio)/2

        if right_eye_ratio>5.3 or left_eye_ratio>5.3:
            eyes_closed="closed"

        else:
            eyes_closed="open"
        
        return eyes_closed
        


def iris_position(iris_center,right_point,left_point,top_point,bottom_point,math):
     
    center_to_right_dist=euclideanDistance(iris_center,right_point,math)
    total_dist_horizontal=euclideanDistance(right_point,left_point,math)
    ratio_horizontal=center_to_right_dist/total_dist_horizontal
    
    center_to_top_dist=euclideanDistance(iris_center,top_point,math)
    total_dist_vertical=euclideanDistance(top_point,bottom_point,math)
    ratio_vertical=center_to_top_dist/total_dist_vertical
    
    iris_position=""
    if ratio_horizontal<=0.38:
        iris_position="right"
        
    elif ratio_horizontal>0.38 and ratio_horizontal<=0.61:
        
        if ratio_vertical<=0.39:
            iris_position="top"
        elif ratio_vertical>0.39 and ratio_vertical<=0.59:
            iris_position="center"
        else:
            iris_position="bottom"
            
        return iris_position,ratio_vertical
        
    else:
        iris_position="left"

    return iris_position,ratio_horizontal

      


    
def main_eye_tracking(frame, face_mesh,cv,np,mp,math,time,statistics):

    # left eyes indices
    LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
    # right eyes indices
    RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 

    # irises Indices list
    RIGHT_IRIS = [474,475, 476, 477]
    LEFT_IRIS = [469, 470, 471, 472]
    L_H_TOP=[159]# right eye
    L_H_BOTTOM=[145]# right eye
    L_H_LEFT = [33]  # right eye right most landmark
    L_H_RIGHT = [133]  # right eye left most landmark

    R_H_LEFT = [362]  # left eye right most landmark
    R_H_RIGHT = [263]  # left eye left most landmark
    R_H_TOP=[386]# left eye
    R_H_BOTTOM=[374]# left eye
    
    
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img_h, img_w = frame.shape[:2]
    results = face_mesh.process(rgb_frame)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    if results.multi_face_landmarks:
        # print((results.multi_face_landmarks[0]))
        # [print(p.x, p.y, p.z ) for p in results.multi_face_landmarks[0].landmark]
            
        mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) 
        for p in results.multi_face_landmarks[0].landmark])

        # cv.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
        # cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
            
        (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
        cv.circle(frame, center_left, int(l_radius), (0,255,0), 2, cv.LINE_AA)
        cv.circle(frame, center_right, int(r_radius), (0,255,0), 2, cv.LINE_AA)

        # cv.circle(frame, center_left, 1, (0,255,0), -1, cv.LINE_AA)
        # cv.circle(frame, center_right, 1, (0,255,0), -1, cv.LINE_AA)
        # drawing on the mask 
        cv.circle(mask, center_left, int(l_radius), (255,255,255), -1, cv.LINE_AA)
        cv.circle(mask, center_right, int(r_radius), (255,255,255), -1, cv.LINE_AA)
            
        #checking if eye is open or not
        eye_status=eyes_closed(mesh_points[L_H_RIGHT],mesh_points[L_H_LEFT],mesh_points[L_H_TOP],mesh_points[L_H_BOTTOM],mesh_points[R_H_RIGHT],mesh_points[R_H_LEFT],mesh_points[R_H_TOP],mesh_points[R_H_BOTTOM],math)
        cv.putText(frame,f"eye status: {eye_status}",(30,50),cv.FONT_HERSHEY_PLAIN,1.2,(0,255,0),1,cv.LINE_AA)

        if eye_status=='open':
            #left
            iris_pos_left,ratio_left=iris_position(center_right,mesh_points[R_H_RIGHT],mesh_points[R_H_LEFT],mesh_points[R_H_TOP],mesh_points[R_H_BOTTOM],math)
            #right
            iris_pos_right,ratio_right=iris_position(center_left,mesh_points[L_H_RIGHT],mesh_points[L_H_LEFT],mesh_points[L_H_TOP],mesh_points[L_H_BOTTOM],math)
            cv.putText(frame,f"Iris Position for right eye {iris_pos_right} {ratio_right:.2f}",(30,60),cv.FONT_HERSHEY_PLAIN,1.2,(0,255,0),1,cv.LINE_AA)
            cv.putText(frame,f"Iris Position for left eye {iris_pos_left} {ratio_left:.2f}",(30,70),cv.FONT_HERSHEY_PLAIN,1.2,(0,255,0),1,cv.LINE_AA)
            return frame,eye_status,iris_pos_right,iris_pos_left
        else:
            return frame,eye_status,'None','None'
    else:
        return frame,'unknown', 'None', 'None'  # Default values when face is not detected
            





# In[ ]:




