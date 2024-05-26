#ECE492_vfedhd_keras.py
#Author : Vyasan Valavil
#Date : 05/26/2024
#This script detects a face in a video stream.
#Press Q to close windows and terminate script

import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition
import time

#It would be ideal to match the accepted webcam specs and you can enter them here:
frameWidth = 640
frameHeight = 480
#This variable holds the video stream as a series of images. 0 represents the default camera.
source = cv2.VideoCapture(0)
#Let's define the framewidth and height to the camera.
source.set(3,frameWidth) #3 is set as width in opencv
source.set(4,frameHeight) #4 is set as the height in opencv

#Now load the model and then the weights (which are obtained after training).
model = model_from_json(open("dataset/facial_expression_model_structure.json","r").read())
model.load_weights('dataset/facial_expression_model_weights.h5')
#The emotions that we are interested in detecting. More are available, but we are forcing the model to pick from these three.
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

#We need a list that will hold all the locations the model detects
all_face_locations = []

#For video, we just process each frame at a time using a while loop.
while True:
    start_time = time.time()
    #Save the current frame by using the read() function. The first variable 'success' is irrelevant
    success,img = source.read()
    #Downsize the image for faster processing times.
    img_small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    #Now detect all the faces in the image #nuber of times to upsample if face is too far from the camera.
    #Model can either be 'hog' or 'cnn'. Hog is much faster although less accurate
    all_face_locations = face_recognition.face_locations(img_small,number_of_times_to_upsample=1,model='hog')
    
    #This for loop will loop through the multiple face locations in the video frame.
    for index,current_face_location in enumerate(all_face_locations):
        # The variable face_location is a tuple which contains the limits of the bounding box that the face is in.
        # We need to split it into individial variables we can work with.
        top,right,bottom,left = current_face_location
        #Change the position maginitude to fit the actual size video frame
        top = top*4
        right = right*4
        bottom = bottom*4
        left = left*4
        #Displaythe location of the face
        print('Found face #{} at top:{},right:{},bottom:{},left:{}'.format(index+1,top,right,bottom,left))
       
        #Extract the current face from the image
        face = img[top:bottom,left:right]
        
        #Draw a bounding Box
        cv2.rectangle(img,(left,top),(right,bottom),(0,0,255),2)
        
        #Need to convert to grayscale
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        #Resize to 48x48 px size
        face = cv2.resize(face, (48, 48))
        #cConvert to an array
        img_pixels = image.img_to_array(face)
        #Convert to a tensor (single row)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        #pCOnvert 0-255 to 0-1
        img_pixels /= 255 
        
        #Perform the prediction and get normalized value for all 7 emotions
        exp_predictions = model.predict(img_pixels) 
        #Find the index with the largest value.
        max_index = np.argmax(exp_predictions[0])
        #Get the corresponding emotion.
        emotion_label = emotions[max_index]
        
        #Display the emotion as text
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, emotion_label, (left,bottom), font, 0.5, (255,255,255),1)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Frame took {duration} seconds to process.")
        
    #Use the 'imshow' command to display each frame in succession:
    cv2.imshow("Video",img)

    # The loop will never end. Even closing the command window will not terminate the script.
    # We need to define a key that will signal the loop to terminate
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#End the stream and close all OpenCV windows.
source.release()
cv2.destroyAllWindows()




