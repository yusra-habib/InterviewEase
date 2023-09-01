#!/usr/bin/env python
# coding: utf-8

# In[5]:


#time ate 2:40 making generalize function
def main_face_count(image,cv2,max_allowed_faces=1):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect all the faces in the image
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)     

    num_faces = len(faces)
    # Check if exactly 1 faces are detected
    if num_faces > max_allowed_faces:
        cv2.putText(image, "more than 1 faces detected.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #returning 2(false) so that it can't go in eye motion tracking file
        return 'two'
    
    elif num_faces==0:
         return 'zero'
    
    else:
        # Display the number of faces on the image
        cv2.putText(image, f"Number of Faces: {num_faces}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return 'one'
    





    

    
    


# In[ ]:




