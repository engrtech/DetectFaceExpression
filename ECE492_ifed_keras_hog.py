import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition
import tensorflow
import time

#Print the version of tensorflow...
print(tensorflow.__version__)
#Load the image and save it as a variable we can work with
img = cv2.imread('images/testing/josepepzlatan.jpg')

#Now load the model and then the weights (which are obtained after training).
model = model_from_json(open("dataset/facial_expression_model_structure.json","r").read())
model.load_weights('dataset/facial_expression_model_weights.h5')
#The emotions that we are interested in detecting. More are available, but we are forcing the model to pick from these three.
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

#Starting timer AFTER the weights have been loaded.
start_time = time.time()

#Now detect all the faces in the image #number of times to upsample if face is too small in the image.
#Model can either be 'hog' or 'cnn'. HOG is much faster although less accurate than a CNN
all_face_locations = face_recognition.face_locations(img,number_of_times_to_upsample=1,model='hog')

#Print the number of faces detected
print('There are {} faces in this image'.format(len(all_face_locations)))

#Looping through the face locations in the image
for index,face_location in enumerate(all_face_locations):
    # The variable face_location is a tuple which contains the limits of the bounding box that the face is in.
    # We need to split it into individual variables we can work with.
    top,right,bottom,left = face_location
    #Printing the location of the current face
    print('Found face #{} at top:{},right:{},bottom:{},left:{}'.format(index+1,top,right,bottom,left))
    #Slicing the current face from the image.
    #Now that we have detected the face, we can ignore the rest of the image.
    face = img[top:bottom,left:right]
    #Draw the bounding box
    cv2.rectangle(img,(left,top),(right,bottom),(0,0,255),2)
    
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
    print(emotion)

    # We can now display the text under the bounding box.
    # HERSHEY_DUPLEX is a common font as it scales without loss of quality and is fast to process.
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, emotion, (left,bottom), font, 0.5, (255,255,255),1)

#Now record the time it took to complete this operation
end_time = time.time()
duration = end_time - start_time

#Use the 'imshow' command to display the image
cv2.imshow("Emotions",img)

print(f"This operation took {duration} seconds to detect a face and it's emotion using Face Recognition.")

#Keep the window open until we press any key
cv2.waitKey(0)
cv2.destroyAllWindows()
