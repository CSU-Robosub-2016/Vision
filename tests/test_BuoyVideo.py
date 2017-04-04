#!/usr/bin/env python
'''
test_simple.py - Tests registration of a single video against a background
                 image, detection and classification of things, and
                 identification of the classified things on screen.
                 '''
import cv2
from vision.vision_tools import VisionTools
from vision.vision_tools import buoyTracker

if __name__ == '__main__':
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("     Buoy Bounding Boxes Test: Takes in a video and         ")
    print("     draws bounding boxes around discovered 'buoys'.        ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Read original video, alter for different videos
    im = cv2.VideoCapture("test_files/Buoy_rightLeft.mov")

    # Define codex and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter("test_files/BuoyRightLeftOut.avi", fourcc, 30.0, (640, 360), True)

    boxes = buoyTracker()

    while(im.isOpened()):
        ret, frame = im.read()
        if not ret:
            break

        # Resize Image
        r = 640 / frame.shape[1]
        dim = (640, int(frame.shape[0] * r))
        image = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        tools = VisionTools()

        final = tools.BuoyBoxes(image, boxes)

        out.write(final)

        cv2.imshow('image', final)
        cv2.waitKey(1)
    im.release()
    out.release()
    cv2.destroyAllWindows()