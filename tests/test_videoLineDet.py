#!/usr/bin/env python
'''
test_simple.py - Tests registration of a single video against a background
                 image, detection and classification of things, and
                 identification of the classified things on screen.
                 '''
from context import vision
import cv2

from vision.cameras.camera_video_feed import videoFeedCamera
from vision.vision_tools import VisionTools


if __name__ == '__main__':
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Line Detection Test: Takes in a video and runs a line detection")
    print("      filtering algorithm on it then displays frame by frame.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    test_cam = videoFeedCamera(
        debug="/home/cskunk/Documents/multicamera_framework/tests/test_files/cam211_trim.mp4")
    tools = VisionTools()

    # Setup a loop to capture a frame from the video feed
    while True:
        final = test_cam.getFrame()
        final = tools.lineDet(final)
        cv2.imshow("Filtered line", final)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
