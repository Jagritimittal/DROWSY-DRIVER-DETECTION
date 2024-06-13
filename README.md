# DROWSY-DRIVER-DETECTION
This project aims to develop such a system to analyze driver behavior in real-time, providing timely alerts and interventions to ensure road safety.In this project we have implemented Image processing technique using eye blink movements and yawning based techniques.
# Methodology
Packages and libraries used in our implementation are :<br>
● Numpy<br>
● Matplotlib<br>
● PIL<br>
● OpenCV<br>
● Dlib<br>
● Scipy<br>
● Imutils<br>
● Pygame<br>

# Implementation 1<br>
Using segmentation and thresholding techniques of image processing to
track whiteness in eyes of the given image
  ## Face Detection <br>
  Face in the given image is segmented and tracked using the haar classifier.
  ## Eyes Detection <br>
  From the segmented image of face eyes are detected using face symmetry and detectMultiscale function of cascade.
  ## Segmenting out eyebrows <br>
  Eyebrows are segmented out of the eyes
  ## Gaussian filter <br>
  Gaussian filter is applied on segmented eye portions to eliminate any additional noise in the frame such as shadows, eyelashes.
  ## Threshold intensity <br>
  Calculated threshold intensity using Characteristic separation adaptive threshold method as follows: -<br>
    a. Considering only first half of image (0-255) as pupil represent lowest value in interval<br>
    b. Replaced zero values with infinity.<br>
    c. Find the maximum value between 0-125<br>
    d. Find the minimum intensity value in the interval of the maximum intensity calculated in above step and 125.<br>
    e. Noted this minimum intensity as threshold intensity.<br>
  ## Thresholding <br>
  Applied thresholding on eye image with above calculated threshold intensity.
  ## Whiteness <br>
  Checked whiteness of eye Counted the number of white and pixels and checked for the whiteness range in the image.
  Whiteness is calculated as number of white pixels divided by total number of pixels
Research paper suggests that the number of white pixels must be at least
5% of total pixels in eyes for an awakened person. So using the same
threshold if whiteness range is below 5% it is classified as drowsy else it
is classified as non drowsy.

# Implementation 2
Using eye blink and yawning movements to detect drowsiness in real
time
  ## Face Detection <br>
  Using dlib’s get_frontal_face_detector to detect
frontal faces in frames
## Facial Landmark Detection 
Using dlib’s shape predictor to predict 68 landmarks on detected face. We used
shape_predictor_68_face_landmarks.dat as the pretrained model for
facial landmark prediction.
The extracted facial landmarks are converted into numpy array and
passed to detect eye and mouth movements.
## Closure of eyes 
Closure/openness of eyes is determined using Eye
Aspect Ratio(EAR). EAR decreases - Drowsiness Increases
EAR = Sum of vertical distance / (2* Horizontal Distance)
## Yawing 
We have used 2 approaches to detect yawning :
First one is using Mouth Aspect Ratio(MAR) which is the ratio of length
and width of the mouth. MAR increases - Drowsiness Increases
Second one is by finding the vertical distance between the upper and
lower lip.<br>
The values obtained are compared with the predefined thresholds.If the
calculated EAR falls below the set threshold for a specific duration or the
MAR or lip distance exceeds the set threshold that means the person is
drowsing.
An auditory alert will play to alert the driver to regain his/her focus.
