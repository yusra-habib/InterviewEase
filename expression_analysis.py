#!/usr/bin/env python
# coding: utf-8

def smile(distanceRatio,middle_between_lips,right_point,left_point,top_point,bottom_point,totaldistance_horiz,totaldistance_vert,math):
    
    center_to_right_dist=euclideanDistance(middle_between_lips,right_point,math)
    #total_dist_horizontal=euclideanDistance(right_point,left_point)
    ratio_horizontal=(center_to_right_dist/totaldistance_horiz)*distanceRatio
    
    center_to_top_dist=euclideanDistance(middle_between_lips,top_point,math)
    
    ratio_vertical=(center_to_top_dist/totaldistance_vert)*distanceRatio
    smile=""
    
    if ratio_vertical>0.19:
        if ratio_horizontal >= 0.52:
            smile = "smile"
        else:
            smile = "neutral"
    else:
        smile="closed lips"

    return smile,ratio_horizontal,ratio_vertical





    
def process_frame(frame, face_mesh,distanceRatio,cv,np,mp,math,time,total_dist_horizontal,total_dist_vertical):

    # lipsUpperOuter indices
    
    lipsUpperOuter =[61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    
    # lipsLowerOuter indices
    lipsLowerOuter=[146, 91, 181, 84, 17, 314, 405, 321, 375, 291] 
    lips_center_upper=[13]
    lips_center_lower=[14]
    Lips_TOP=[0]# Upper top
    Lips_BOTTOM=[17]# Upper bottom
    
    Lips_LEFT = [291]  # Upper right most landmark
    Lips_RIGHT = [61]  # Upper left most landmark

    
    
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img_h, img_w = frame.shape[:2]
    results = face_mesh.process(rgb_frame)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    if results.multi_face_landmarks:
        # print((results.multi_face_landmarks[0]))
        # [print(p.x, p.y, p.z ) for p in results.multi_face_landmarks[0].landmark]
            
        mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) 
        for p in results.multi_face_landmarks[0].landmark])
        
        middle_between_lips_x = int((mesh_points[Lips_LEFT][0][0] + mesh_points[Lips_RIGHT][0][0]) / 2)
        middle_between_lips_y = int((mesh_points[Lips_TOP][0][1] + mesh_points[Lips_BOTTOM][0][1]) / 2)
        middle_between_lips = [[middle_between_lips_x, middle_between_lips_y]]
        middle_between_lips=np.array(middle_between_lips)
        middle_between_lips1 = [middle_between_lips_x, middle_between_lips_y]


        # Draw dots on lips landmarks
        for point in mesh_points[lipsUpperOuter]:
            cv.circle(frame, tuple(point), 3, (0, 255, 0), -1)  # Green dots for upper outer lips
        for point in mesh_points[lipsLowerOuter]:
            cv.circle(frame, tuple(point), 3, (0, 0, 255), -1)  # Red dots for lower outer lips
        cv.circle(frame, tuple(middle_between_lips1), 3, (255, 0, 0), -1)  # Blue dot for lips center


        #lips
        
        smiling='None'
        smiling,ratio_h,ratio_v=smile(distanceRatio,middle_between_lips,mesh_points[Lips_RIGHT],mesh_points[Lips_LEFT],mesh_points[Lips_TOP],mesh_points[Lips_BOTTOM],total_dist_horizontal,total_dist_vertical,math)
        cv.putText(frame,f"{smiling} {ratio_h:.2f} {ratio_v:.2f}",(30,80),cv.FONT_HERSHEY_PLAIN,1.2,(0,255,0),1,cv.LINE_AA)
        
    return frame,mask,smiling

def euclideanDistance(point1,point2,math):
    x1,y1=point1.ravel()
    x2,y2=point2.ravel()
    distance=math.sqrt(((x2-x1)**2)+((y2-y1)**2))
    return distance


def smileImage(middle_between_lips,right_point,left_point,top_point,bottom_point,math): 
    
    
    center_to_right_dist=euclideanDistance(middle_between_lips,right_point,math)
    total_dist_horizontal=euclideanDistance(right_point,left_point,math)
    ratio_horizontal=center_to_right_dist/total_dist_horizontal
    
    total_dist_vertical=euclideanDistance(left_point,top_point,math)
    center_to_top_dist=euclideanDistance(middle_between_lips,top_point,math)
    ratio_vertical=center_to_top_dist/total_dist_vertical
    smile=""
    
    if ratio_vertical>0.19:
        if ratio_horizontal>= 0.52:
            smile = "smile"
        else:
            smile = "neutral"
    else:
        smile="closed lips"

    return smile,ratio_horizontal,ratio_vertical,total_dist_horizontal,total_dist_vertical

def process_frameImage(frame, face_mesh,cv,np,mp,math,time):
    # lipsUpperOuter indices
    lipsUpperOuter =[61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291] 
    # lipsLowerOuter indices
    lipsLowerOuter=[146, 91, 181, 84, 17, 314, 405, 321, 375, 291] 
    lips_center_upper=[13]
    lips_center_lower=[14]
    Lips_TOP=[0]# Upper top
    Lips_BOTTOM=[17]# Upper bottom
    Lips_LEFT = [291]  # Upper right most landmark
    Lips_RIGHT = [61]  # Upper left most landmark
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img_h, img_w = frame.shape[:2]
    results = face_mesh.process(rgb_frame)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    if results.multi_face_landmarks:
   
        mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) 
        for p in results.multi_face_landmarks[0].landmark])
        
        middle_between_lips_x = int((mesh_points[Lips_LEFT][0][0] + mesh_points[Lips_RIGHT][0][0]) / 2)
        middle_between_lips_y = int((mesh_points[Lips_TOP][0][1] + mesh_points[Lips_BOTTOM][0][1]) / 2)
        middle_between_lips = [[middle_between_lips_x, middle_between_lips_y]]
        middle_between_lips=np.array(middle_between_lips)
        middle_between_lips1 = [middle_between_lips_x, middle_between_lips_y]


        # Draw dots on lips landmarks
        for point in mesh_points[lipsUpperOuter]:
            cv.circle(frame, tuple(point), 3, (0, 255, 0), -1)  # Green dots for upper outer lips
        for point in mesh_points[lipsLowerOuter]:
            cv.circle(frame, tuple(point), 3, (0, 0, 255), -1)  # Red dots for lower outer lips
        cv.circle(frame, tuple(middle_between_lips1), 3, (255, 0, 0), -1)  # Blue dot for lips center

        #lips
        
        
        smile,ratio_horizontal,ratio_vertical,total_dist_horizontal,total_dist_vertical=smileImage(middle_between_lips,mesh_points[Lips_RIGHT],mesh_points[Lips_LEFT],mesh_points[Lips_TOP],mesh_points[Lips_BOTTOM],math)
        #cv.putText(frame,f"{smile} {ratio:.2f}",(30,50),cv.FONT_HERSHEY_PLAIN,1.2,(0,255,0),1,cv.LINE_AA)
        
    return frame,mask,total_dist_horizontal,total_dist_vertical


def main_expression_analysis(total_dist_horizontal,total_dist_vertical,distance,cv,np,mp,math,time,face_mesh,frame):
    from depth_estimation import process_depth_estimation,depth_estimation
    #print(distance)
    distanceNew=process_depth_estimation(frame,face_mesh,cv,mp)
    ditanceRatio=distanceNew/distance
    frame, mask,expression_value = process_frame(frame, face_mesh,ditanceRatio,cv,np,mp,math,time,total_dist_horizontal,total_dist_vertical)
    return frame,expression_value

                    
                
    

    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




