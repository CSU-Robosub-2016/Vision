#!/usr/bin/env python
'''
test_simple.py - Tests registration of a single video against a background
                 image, detection and classification of things, and
                 identification of the classified things on screen.
                 '''
import cv2
from vision.vision_tools import VisionTools

if __name__ == '__main__':
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("     Buoy Bounding Boxes Test: Takes in a video and         ")
    print("     draws bounding boxes around discovered 'buoys'.        ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Read original video, alter for different videos
    im = cv2.VideoCapture("test_files/Buoy_Still.mov")

    # Define codex and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter("test_files/BuoyVideoOut.avi", fourcc, 60.0, (1920, 1080), False)

    while(im.isOpened()):
        ret, frame = im.read()
        if not ret:
            break
        im2 = cv2.medianBlur(frame, 5)

        tools = VisionTools()

        final = tools.BuoyBoxes(im2)

        cv2.imshow('image', final)
        cv2.waitKey(5)
