from flask import Flask, render_template, Response,request, jsonify,session,redirect, url_for
import numpy as np
import cv2 as cv 
import mediapipe as mp 
import math as math
import time as time
#import requests
import statistics as statistics 
import pyaudio,wave




from eye_tracking import iris_position,eyes_closed,euclideanDistance,main_eye_tracking
from expression_analysis import process_frame,smile,smileImage,euclideanDistance,process_frameImage,main_expression_analysis
from face_count import main_face_count
from depth_estimation import process_depth_estimation,depth_estimation
#from eyebrows_analysis import *

app = Flask(__name__, template_folder='templates')
app.secret_key = 'yusraHabib'


camera = None
start_Video_time=0
ConfidenceLevel=0
audio = None
audio_stream = None
audio_frames = []
recording = False
confidentScore=0
frame_saved=None
ConfidenceLevel='not found'
capture_frame_requested=True
eyeStatus_values_after_3seconds = []
eyeTracking_values_after_3seconds = []   
expression_values_after_3seconds = []
number_faces_after_3seconds=[]






def saveImage():   
    
   # Load the captured and saved image
   saved_image_path = "thirty_frame.jpg"
   frame = cv.imread(saved_image_path)
   return frame

def scores_calc(total_score,eyeTracking_values_after_3seconds,eyeStatus_values_after_3seconds,number_faces_after_3seconds,expression_values_after_3seconds): 
    eyeTracking_scores_after_3seconds = []
    eyeStatus_scores_after_3seconds = []
    expression_scores_after_3seconds = []
    number_faces_scores_after_3seconds = [] 


    global ConfidenceLevel
    scores = {'open': 3, 'closed': 1, 'one': 2, 'two': 0, 'zero': 0, 'closed lips': 1, 'smile': 3, 'neutral': 2, 'center': 3, 'top': 1, 'right': 1, 'left': 1, 'bottom': 1, 'None': 0}
    total_sum = 0
    for i in range(len(eyeTracking_values_after_3seconds)):
        eyeTracking_scores = []
        for j in range(len(eyeTracking_values_after_3seconds[i])):
            value = scores.get(eyeTracking_values_after_3seconds[i][j], 0)
            eyeTracking_scores.append(value)
            total_sum += value
        eyeTracking_scores_after_3seconds.append(eyeTracking_scores)

    for i in range(len(eyeStatus_values_after_3seconds)):
        value = scores.get(eyeStatus_values_after_3seconds[i], 0)
        eyeStatus_scores_after_3seconds.append(value)
        total_sum += value

    for i in range(len(expression_values_after_3seconds)):
        value = scores.get(expression_values_after_3seconds[i], 0)
        expression_scores_after_3seconds.append(value)
        total_sum += value

    for i in range(len(number_faces_after_3seconds)):
        value = scores.get(number_faces_after_3seconds[i], 0)
        number_faces_scores_after_3seconds.append(value)
        total_sum += value

    score=total_sum
    confidentScore=score/total_score

    #highest score 120/40=3 
    if confidentScore>2.5 and confidentScore<=3:
        ConfidenceLevel='high'
        #print(f"Confidence level is high and confidence score is {confidentScore}")
    elif confidentScore>=1.5 and confidentScore<=2.5:
        ConfidenceLevel='medium'
        #print(f"Confidence level is medium and confidence score is {confidentScore}")
    elif confidentScore>=0 and confidentScore<1.5:
        ConfidenceLevel='low'
        #print(f"Confidence level is low and confidence score is {confidentScore}")
    else:
        ConfidenceLevel='not found'
        #print('cannot calculate confidence score')
    return ConfidenceLevel,confidentScore


def generate_frames():
    global camera
    global start_Video_time
    global capture_frame_requested
    #global frame_saved
    #initialize variable
    eyeStatus_values = []
    eyeTracking_valuesRight = []
    eyeTracking_valuesLeft = []
    expression_values = []
    number_faces=[]
        

        

    update_interval = 4#seconds
    start_time = time.time()
    saveImag_counter=0
    frame_saved=saveImage()
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
            #print('ready!!')
                    
        #frame1=frame
        distance=process_depth_estimation(frame_saved,face_mesh,cv,mp)
        
        frame_saved,mask1,total_dist_horizontal,total_dist_vertical = process_frameImage(frame_saved, face_mesh,cv,np,mp,math,time)
        #frame_counter=0
        start_Video_time = time.time()
        while True:
            if camera is not None:
                success, frame = camera.read()
                frame = cv.flip(frame, 1)
                if not success:
                    break
                #frame = cv.equalizeHist(frame)
                """ ret, buffer = cv.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
                    """

                if (capture_frame_requested==True):

                    #print('distance',distance)
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
                    
                    # Convert the frame to JPEG format
                    #dframe = cv.GaussianBlur(frame, (5, 5), 0)
                    _, buffer = cv.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()

                    # Yield the frame to the client
                    yield (b'--frame\r\n'b'Content-Type: imagjpeg\r\n\r\n' + frame_bytes + b'\r\n')


                else:
                    print('detection is not start')
                    _, buffer = cv.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()

                    # Yield the frame to the client
                    yield (b'--frame\r\n'b'Content-Type: imagjpeg\r\n\r\n' + frame_bytes + b'\r\n')


def generate_framesImage():
    while True:
        if camera is not None:
            success, frame = camera.read()
            frame = cv.flip(frame, 1)
            if not success:
                break
            #frame = cv.equalizeHist(frame)
            _, buffer = cv.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame to the client
            yield (b'--frame\r\n'b'Content-Type: imagjpeg\r\n\r\n' + frame_bytes + b'\r\n')

def start_audio_recording():
    global audio_stream
    global audio
    if audio is None:
        audio = pyaudio.PyAudio()
        audio_stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024
        )
        return audio_stream

def stop_audio_recording():
    global audio_stream
    global audio
    print('audio_stream',audio_stream)
    if audio_stream is not None:
        audio_stream.stop_stream()
        audio_stream.close()
        audio.terminate()
        audio_stream = None
        audio = None

def save_audio_to_file():
    global audio_frames
    print('audio_stream',audio_stream)
    if audio_frames:
        audio_data = b''.join(audio_frames)
        with wave.open('audio.wav', 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(audio_data)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/startInterview')  # Use the correct endpoint name
def startInterview():  # Use the correct function name
    return render_template('startInterview.html')

@app.route('/videoAnalysis')  # Use the correct endpoint name
def videoAnalysis():  # Use the correct function name
    return render_template('videoAnalysis.html')

@app.route('/interviewQuestion')  
def interviewQuestion():  
    return render_template('interviewQuestion.html', last_clicked_question=session.get('last_clicked_question'))


@app.route('/start_camera')
def start_camera():
    global camera
    global audio_stream
    if camera is None:
        camera = cv.VideoCapture(0)
        audio_stream = start_audio_recording()
        return 'Camera and audio recording started'  # Open the webcam (0 indicates default camera)

    return 'Camera started'

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feedImage')
def video_feedImage():
    return Response(generate_framesImage(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    global camera
    global audio_frames
    global recording

    if camera is not None:
        camera.release()
        stop_audio_recording()
        camera = None
        save_audio_to_file()  # Save audio to a file
        audio_frames = []  # Clear the audio frames list
        recording = False
        return 'Camera stopped and audio saved to file (audio.wav)'
    else:
        return 'Camera is already stopped'
    
@app.route('/result')
def result():
    global ConfidenceLevel 
    global confidentScore
    #allQuestionTime=[]
    
    print(expression_values_after_3seconds)
    current_Video_time = time.time()
    total_Video_time=current_Video_time-start_Video_time
    #total_Video_time = math.floor(total_Video_time/4)
    #allQuestionTime.append(total_Video_time)
    #total_score=total_Video_time*5
    if len(number_faces_after_3seconds)!=0:
        total_score=len(number_faces_after_3seconds)*3*2
        if total_Video_time>4:
            ConfidenceLevel,confidentScore=scores_calc(total_score,eyeTracking_values_after_3seconds,eyeStatus_values_after_3seconds,number_faces_after_3seconds,expression_values_after_3seconds)
                    
        else:
            print('some problem occur,may be total video time is not greter than 4 sec') 
    
    return render_template('result.html', ConfidenceLevel=ConfidenceLevel, confidentScore=confidentScore)

@app.route('/capture_frame')
def capture_frame():
    global capture_frame_requested
    global frame_saved
    success, frame = camera.read()  # Capture a frame from the camera
    if success:
        frame_saved = cv.flip(frame, 1)
        #numberface=main_face_count(frame_saved,cv,1)
        #print(numberface)
        capture_frame_requested=True
        cv.imwrite("frame_saved.jpg", frame_saved)
        return stop_camera()


@app.route('/update_session', methods=['POST'])
def update_session():
    question_id = request.json.get('questionId')
    if question_id is not None:
        answered_questions = session.get('answered_questions', [])
        if question_id not in answered_questions:
            answered_questions.append(question_id)
            session['answered_questions'] = answered_questions
            session['last_clicked_question'] = question_id  # Store the last clicked question
    return jsonify(success=True)


if __name__ == '__main__':
    app.run(debug=True)
            

