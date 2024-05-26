#ECE492_ifd_dlib
#Author : Vyasan Valavil
#Date : 05/26/2024
#This script detects a face using DLib (which uses HOG)
#Press any key to stop the script and close image window

import cv2
import dlib
import time

#Load an image to detect and save it as a variable we can work with.
img = cv2.imread('images/testing/zlatanpepjose.jpg')

start_time = time.time()

#The dlib HOG detector works with grayscale so we need to convert our image to it.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Now load the model. This is a pretrained model.
face_detection_classifier = dlib.get_frontal_face_detector()

#Detect all face locations using the HOG classifier
all_face_locations = face_detection_classifier(img_gray,1)

#print the number of faces detected
print('There are {} faces in this image'.format(len(all_face_locations)))

#This for loop will loop through the multiple face locations in the image
for index,face_location in enumerate(all_face_locations):
    # The variable face_location is a tuple which contains the limits of the bounding box that the face is in.
    # We need to split it into individual variables we can work with.
    left, top, right, bottom = face_location.left(),face_location.top(),face_location.right(),face_location.bottom()
    #Printing the location of the current face
    print('Found face #{} at top:{},right:{},bottom:{},left:{}'.format(index+1,top,right,bottom,left))
    #Draw the bounding box
    cv2.rectangle(img,(left,top),(right,bottom),(0,0,255),2)

#Now record the time it took to complete this operation
end_time = time.time()
duration = end_time - start_time
print(f"This operation took {duration} seconds to detect a face using DLib.")

#Show the image
cv2.imshow("Faces in Image",img)

#Keep the window open until we press any key
cv2.waitKey(0)
cv2.destroyAllWindows()
