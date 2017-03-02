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
    print("Buoy Bounding Boxes Test: Takes in a single image and ")
    print("     draws bounding boxes around discovered 'buoys'.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    image = cv2.imread("test_files/Circle.png")

    # Resize Image
    r = 640.0 / image.shape[1]
    dim = (640, int(image.shape[0] * r))

    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('image', image)
    cv2.waitKey(0)

    tools = VisionTools()

    final = tools.BuoyBoxes(image)
    cv2.imshow('image', image)
    cv2.waitKey(0)
