#ECE492_ifd_hog
#Author : Vyasan Valavil
#Date : 05/26/2024
#This script detects a face using HOG
#Press any key to stop the script and close image window

import cv2
import face_recognition
import time

#Load an image to detect and save it as a variable we can work with.
img = cv2.imread('images/testing/zlatanpepjose.jpg')

start_time = time.time()

#Detect all face locations using the HOG classifier
all_face_locations = face_recognition.face_locations(img,model='hog')

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
    cv2.rectangle(img,(left,top),(right,bottom),(0,0,255),2)

#Now record the time it took to complete this operation
end_time = time.time()
duration = end_time - start_time
print(f"This operation took {duration} seconds to detect a face using Face Recognition.")

#Show the image
cv2.imshow("Faces in Image",img)

#Keep the window open until we press any key
cv2.waitKey(0)
cv2.destroyAllWindows()
