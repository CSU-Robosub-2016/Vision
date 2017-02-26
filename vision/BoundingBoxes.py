import argparse
import imutils
import cv2
import numpy as np
import sys

print(cv2.__version__)

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=False, help="Path to the input image")
# args = vars(ap.parse_args())

image = cv2.imread(sys.argv[1])

# Resize Image
r = 640.0 / image.shape[1]
dim = (640, int(image.shape[0] * r))

image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('image', image)
cv2.waitKey(0)

# Get average background color
avg_color_rows = np.average(image, axis=0)
avg_color = np.average(avg_color_rows, axis=0)
avg_color = np.uint8(avg_color)


# Create image of average color (no use in final product)
avg_color_img = np.array([[avg_color]*640]*360, np.uint8)

cv2.imshow('Average Image Color', avg_color_img)
cv2.waitKey(0)

# Filter out background avg color
# Part a: filter out below avg-10
# upper = [0,0,0]
# for i in range (0, 3):
#     if avg_color[i] - 10 > 0:
#         upper[i] = avg_color[i] - 10

upper_filter = np.array([avg_color[0] + 10, avg_color[1] + 10, avg_color[2] + 10])
lower_filter = np.array([avg_color[0] - 10, avg_color[1] - 10, avg_color[2] - 10])

mask = cv2.inRange(image, lower_filter, upper_filter)
invert = cv2.bitwise_not(mask)
res = cv2.bitwise_and(image, image, mask=invert)

cv2.imshow('frame', image)
cv2.imshow('mask', mask)
cv2.imshow('invert', invert)
cv2.imshow('res', res)

cv2.waitKey(0)

frame = cv2.medianBlur(res, 5)

# Threshold filter to find contours
ret, thresh = cv2.threshold(frame, 153, 255, 0)
bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Find Contours
q, contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create Rectangle around largest contour (ROI)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 2)

cv2.imshow('final', image)
cv2.waitKey(0)