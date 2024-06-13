# -*- coding: utf-8 -*-
"""Drowsy using segmentation and thresholding.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1n20WcVSXvh9OESYwtgGubD1xLZMl1bZy
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install opencv-python

import pandas as pd

import os

import cv2
import numpy
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow

import cv2
import numpy as np


# init part
face_cascade = cv2.CascadeClassifier('/content/drive/MyDrive/Eye-Tracker-No_GUI/Eye-Tracker-No_GUI/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/content/drive/MyDrive/Eye-Tracker-No_GUI/Eye-Tracker-No_GUI/haarcascade_eye.xml')


def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
    return frame


def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]
    return left_eye, right_eye


def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)

    return img


def nothing(x):
    pass
frame = cv2.imread("/content/drive/MyDrive/Eye-Tracker-No_GUI/Eye-Tracker-No_GUI/image.jpg")
cv2_imshow(frame)

"""##Detecting face in given image"""

face_frame = detect_faces(frame, face_cascade)
cv2_imshow(face_frame)

"""##Detecting Eyes in given image"""

eyes = detect_eyes(face_frame, eye_cascade)

for eye in eyes:
    if eye is not None:
      print("Eyes detected : ")
      cv2_imshow(eye)

"""##Eyes After segmenting out the eyebrows"""

for eye in eyes:
    if eye is not None:
      print("Eyes after segmenting the eyebrows : ")
      eye = cut_eyebrows(eye)
      cv2_imshow(eye)

"""##Applying guassian filter"""

for eye in eyes:
    if eye is not None:
      print("Eyes after applying guassian filter : ")
      eye = cut_eyebrows(eye)
      eye = cv2.GaussianBlur(eye, (5, 5), 0)
      cv2_imshow(eye)

"""##Plotting histogram of eyes before thresholding"""

def ht(im):
  cv2_imshow(im)
  hist,bins = np.histogram(im,bins=256,range=[0,256])
  plt.bar(bins[:-1],hist,width=1)
  plt.xlabel("Intensity Values")
  plt.ylabel("Frequency")
  plt.show()

for eye in eyes:
    if eye is not None:
      eye = cut_eyebrows(eye)
      filter_size = int(0.1 * min(eye.shape[:2]))
      filter_size = filter_size if filter_size % 2 == 1 else filter_size + 1
      eye = cv2.GaussianBlur(eye, (filter_size, filter_size), 0)
      ht(eye)

"""##Converting eyes to gray scale image and calculated threshold intensity"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_smoothing_filter(eye):
    # Calculate the size of the smoothing filter based on 10% of the eye size
    filter_size = max(1, int(0.1 * min(eye.shape[:2])))

    # Ensure filter_size is odd
    filter_size = filter_size if filter_size % 2 == 1 else filter_size + 1

    # Apply Gaussian blur
    eye = cv2.GaussianBlur(eye, (filter_size, filter_size), 0)

    return eye

# Assuming 'eyes' is a list containing the eye images
for eye in eyes:
    if eye is not None:
        eye = cut_eyebrows(eye)  # Assuming cut_eyebrows is a function to remove eyebrows
        eye = apply_smoothing_filter(eye)
        eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if necessary
        cv2_imshow(eye_gray)
        # Calculate histogram
        hist = cv2.calcHist([eye_gray], [0], None, [256], [0, 256])

        # Replace zeros with infinite
        hist[hist == 0] = np.inf

        # Find peak values (peaks are the local maxima) within the range of 0 to 125
        peaks = np.where((hist[:126] < hist[1:127]) & (hist[1:127] > hist[2:128]))[0] + 1

        # Filter out infinity from peak values
        non_inf_peaks = [peak for peak in peaks if hist[int(peak)] != np.inf]
        # Find the maximum peak value within the range of 0 to 125
        max_peak_value = np.max(hist[non_inf_peaks])
        # Find the index of the maximum peak value
        max_peak_index = np.argmax(hist == max_peak_value)

        # Find the index of the minimum value between the maximum peak index and 125
        threshold_index = np.argmin(hist[max_peak_index:126]) + max_peak_index
        print("Threshold_intensity",threshold_index)
        # Draw histogram
        plt.plot(hist)
        plt.axvline(x=threshold_index, color='r', linestyle='--', label='Threshold')
        plt.legend()
        plt.title('Histogram')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.show()

"""##Applying thresholding and calculating whiteness in image"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_smoothing_filter(eye):
    # Calculate the size of the smoothing filter based on 10% of the eye size
    filter_size = max(1, int(0.1 * min(eye.shape[:2])))

    # Ensure filter_size is odd
    filter_size = filter_size if filter_size % 2 == 1 else filter_size + 1

    # Apply Gaussian blur
    eye = cv2.GaussianBlur(eye, (filter_size, filter_size), 0)

    return eye

# Assuming 'eyes' is a list containing the eye images
for eye in eyes:
    if eye is not None:
        eye = cut_eyebrows(eye)  # Assuming cut_eyebrows is a function to remove eyebrows
        eye = apply_smoothing_filter(eye)
        eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if necessary

        # Calculate histogram
        hist = cv2.calcHist([eye_gray], [0], None, [256], [0, 256])

        # Replace zeros with infinite
        hist[hist == 0] = np.inf

        # Find peak values (peaks are the local maxima) within the range of 0 to 125
        peaks = np.where((hist[:126] < hist[1:127]) & (hist[1:127] > hist[2:128]))[0] + 1

        # Filter out infinity from peak values
        non_inf_peaks = [peak for peak in peaks if hist[int(peak)] != np.inf]

        # Find the maximum peak value within the range of 0 to 125
        max_peak_value = np.max(hist[non_inf_peaks])

        # Find the index of the maximum peak value
        max_peak_index = np.argmax(hist == max_peak_value)

        # Find the index of the minimum value between the maximum peak index and 125
        threshold_index = np.argmin(hist[max_peak_index:126]) + max_peak_index

        # Apply thresholding
        _, thresholded_eye = cv2.threshold(eye_gray, threshold_index, 255, cv2.THRESH_BINARY)

        ht(thresholded_eye)
        z = np.count_nonzero(thresholded_eye==0)
        print("No. of Black pixels ",z)
        nz= np.count_nonzero(thresholded_eye)
        print("No. of white pixels ",nz)
        print("Ratio of white pixels to total pixels is : ",nz/(z+nz))
        if((nz>((5*(z+nz)/100)))&(nz<((50*(z+nz)/100)))):
          print("Non-drowsy")
        else :
          print("Drowsy")