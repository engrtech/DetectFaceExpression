#ECE492_ifed_keras_cnn_callable.py
#Author : Vyasan Valavil
#Date : 05/26/2024
#This script will be called by the function in ECE492_call_ifed_keras.py
#This script will expect two arguments for each call
#First argument is the location of the image
#Second arg

import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition
import tensorflow
import time
import sys
import os

#gather the arguments as a string and
img_loc = sys.argv[1]
hog_or_cnn = sys.argv[2]

#Load the image and save it as a variable we can work with
img = cv2.imread(img_loc)

#Now load the model and then the weights (which are obtained after training).
model = model_from_json(open("dataset/facial_expression_model_structure.json","r").read())
model.load_weights('dataset/facial_expression_model_weights.h5')
#The emotions that we are interested in detecting. More are available, but we are forcing the model to pick from these three.
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

#Starting timer AFTER the weights have been loaded.
start_time = time.time()

#Now detect all the faces in the image #number of times to upsample if face is too small in the image.
#Model can either be 'hog' or 'cnn'. HOG is much faster although less accurate than a CNN
all_face_locations = face_recognition.face_locations(img,number_of_times_to_upsample=1,model=hog_or_cnn)
#Initialize an area because we are only going to analyze the largest face
max_area = 0

if len(all_face_locations)==0:
    print("No Faces")
    sys.exit()

#Looping through the face locations in the image
for index,face_location in enumerate(all_face_locations):
    # The variable face_location is a tuple which contains the limits of the bounding box that the face is in.
    # We need to split it into individual variables we can work with.
    top,right,bottom,left = face_location
    #Let's calculate the pizels inside the frame around the face
    face_area = int(bottom-top)*int(right-left)
    if face_area>max_area:
        max_area = face_area
        #Slicing the current face from the image.
        #Now that we have detected the face, we can ignore the rest of the image.
        face = img[top:bottom,left:right]
    
#Convert the face to grayscale so we can perform the convulutions.
face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#Resize to 48 x 48 pixels
face = cv2.resize(face, (48, 48))
#Convert to a numpy array
img_pixels = image.img_to_array(face)
#Convert to a tensor (A single row array)
img_pixels = np.expand_dims(img_pixels, axis = 0)
#CNN works with normalized values. COnvert 0-255 to 0-1
img_pixels /= 255

#Utilize the model to do predictions.
exp_predictions = model.predict(img_pixels)
#Of all the outputs int he outer layer, we select the highest amount.
max_index = np.argmax(exp_predictions[0])
#Get the corresponding matching emotion from the list.
emotion = emotions[max_index]
print(f'Emotion: {emotion}')
exit()
