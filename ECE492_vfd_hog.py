#ECE492_vfd_hog.py
#Author : Vyasan Valavil
#Date : 05/26/2024
#This script detects a face in a video stream
#Press Q to close windows and terminate script

import cv2
import face_recognition

#It would be ideal to match the accepted webcam specs and you can enter them here:
frameWidth = 640
frameHeight = 480
#This variable holds the video stream as a series of images. 0 represents the default camera.
source = cv2.VideoCapture(0)
#Let's define the framewidth and height to the camera.
source.set(3,frameWidth) #3 is set as width in opencv
source.set(4,frameHeight) #4 is set as the height in opencv


#Initialize the array variable to hold all face locations in the frame
all_face_locations = []

#For video, we just process each frame at a time using a while loop.
while True:
    #Save the current frame by using the read() function. The first variable 'success' is irrelevant
    success,img = source.read()
    #Downsize the image for faster processing times.
    img_small = cv2.resize(img,(0,0),fx=0.25,fy=0.25)
    #Now detect all the faces in the image #nuber of times to upsample if face is too far from the camera.
    #Model can either be 'hog' or 'cnn'. Hog is much faster although less accurate
    all_face_locations = face_recognition.face_locations(img_small,number_of_times_to_upsample=1,model='hog')
    
    #This for loop will loop through the multiple face locations in the video frame.
    for index,face_location in enumerate(all_face_locations):
        #The variable face_location is a tuple which contains the limits of the bounding box that the face is in.
        #We need to split it into individial variables we can work with.
        top,right,bottom,left = face_location
        #Since we downsized the image earlier, we need to upsize the numbers so they can be mapped onto the size of the video stream.
        top = top*4
        right = right*4
        bottom = bottom*4
        left = left*4
        #We can display the location of where the face is located in the output window if needed...
        print('Found face #{} at top:{},right:{},bottom:{},left:{}'.format(index+1,top,right,bottom,left))
        #Draw the bounding box
        cv2.rectangle(img,(left,top),(right,bottom),(0,0,255),2)
    #Use the 'imshow' command to display each frame in succession:
    cv2.imshow("Webcam Video",img)

    #The loop will never end. Even closing the command window will not terminate the script.
    #We need to define a key that will signal the loop to terminate
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#End the stream and close all OpenCV windows.
source.release()
cv2.destroyAllWindows()        










