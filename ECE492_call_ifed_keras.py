#ECE492_call_ifed_keras.py
#Author : Vyasan Valavil
#Date : 05/26/2024
#This script will utilize a callable script. Location is defined in the variable script_path
############################################################################################
import subprocess
import time

#The next two variables need to be updated depending on where the callable script is
#in reference to this script
#Enter the location of the script that needs to be called
script_path = 'ECE492_ifed_keras_cnn_callable.py'
#Enter the model to be used (HOG or CNN)
hog_or_cnn = 'hog'

#The following function runs a subprocess does performs the face detection.
def get_emotion(image_loc):
    global script_path, hog_or_cnn
    arguments = [image_loc, hog_or_cnn] 
    command = ['python3', script_path] + arguments
    try:
        result = subprocess.run(
        command,  # Command to execute
        stdout=subprocess.PIPE,  # Capture standard output
        stderr=subprocess.PIPE,  # Capture standard errors
        text=True  # Output will be treated as text (Python 3.7+);
        )
        lines = result.stdout.split('\n') #split the stdouts by line
        return lines[-2] #The last line is always ' ' so 2nd last line is the output emotion
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while trying to run the script: {e}")
 
##########################################################################################

#The following examples can be changed and instead of print,
#the results can also be written to a file if needed.
image_loc = 'images/testing/zlatanpepjose.jpg'
print(f'Emotion displayed in {image_loc} is {get_emotion(image_loc)}')

image_loc = 'images/testing/sadwoman.jpg'
print(f'Emotion displayed in {image_loc} is {get_emotion(image_loc)}')

image_loc = 'images/testing/disgustedman.jpg'
print(f'Emotion displayed in {image_loc} is {get_emotion(image_loc)}')
