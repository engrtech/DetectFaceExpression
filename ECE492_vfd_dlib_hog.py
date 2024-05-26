import cv2
import dlib

#It would be ideal to match the accepted webcam specs and you can enter them here:
frameWidth = 640
frameHeight = 480
#This variable holds the video stream as a series of images. 0 represents the default camera.
source = cv2.VideoCapture(0)
#Let's define the framewidth and height to the camera.
source.set(3,frameWidth) #3 is set as width in opencv
source.set(4,frameHeight) #4 is set as the height in opencv

#Now load the model. This is a pre-trained model
face_detection_classifier = dlib.get_frontal_face_detector()

#We need a list that will hold all the locations the model detects
all_face_locations = []

#For video, we just process each frame at a time using a while loop.
while True:
    #Save the current frame by using the read() function. The first variable 'success' is irrelevant
    success,img = source.read()
    #The dlib HOG detector works with grayscale so we need to convert our image to it.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Downsize the image for faster processing times.
    img_gray_small = cv2.resize(img_gray,(0,0),fx=0.25,fy=0.25)
    #Detect all face locations using the HOG classifier
    all_face_locations = face_detection_classifier(img_gray_small,1)
    #This for loop will loop through the multiple face locations in the video frame.
    for index,face_location in enumerate(all_face_locations):
        #The variable face_location is a tuple which contains the limits of the bounding box that the face is in.
        #We need to split it into individual variables we can work with.
        left, top, right, bottom = face_location.left(),face_location.top(),face_location.right(),face_location.bottom()
        #change the position maginitude to fit the actual size video frame
        top = top*4
        right = right*4
        bottom = bottom*4
        left = left*4
        #Printing the location of the current face
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
