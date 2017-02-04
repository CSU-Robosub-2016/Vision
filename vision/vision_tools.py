'''
vision_tools.py - Tools for the vision portion of the RoboSub project.
Contains useful filters and processing routines for detection and
target tracking.
'''
import numpy as np
import cv2


class VisionTools:
    ##
    # @brief Does nothing important for the time being.  Will add some
    #        functionality later
    def __init__(self):
        # Dummy variable, not used for anything
        self.useme = True

    ##
    # @brief Orange line detection, developed by Brett Gonzales
    # @param frame The frame to be filtered
    # @return output Filtered frame of largest contour
    def lineDet(self, frame):
        frame = cv2.medianBlur(frame, 5)

        # Threshold filter to find contours
        ret, thresh = cv2.threshold(frame, 153, 255, 0)
        bw = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        # Find Contours
        q, contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)

        # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]

        # Create Rectangle around largest contour (ROI)
        x, y, w, h = cv2.boundingRect(cnt)
        new = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Compare region of interest to original image,
        #  set everything outside the ROI to zero
        mask = np.zeros(new.shape, np.uint8)
        mask[y:y+h, x:x+w] = new[y:y+h, x:x+w]
        out = cv2.bitwise_and(new, mask)

        # Filter for Max RGB values
        b, g, r = cv2.split(out)
        M = np.maximum(np.maximum(r, g), b)
        r[r < M] = 0
        g[g < M] = 0
        b[b < M] = 0

        # Merge back into a maximum RGB image
        image2 = cv2.merge([b, g, r])

        # Set upper and lower limits for colors
        lower = np.array([0, 0, 10])
        upper = np.array([0, 0, 255])

        # Filter out all colors except the tape
        mask = cv2.inRange(image2, lower, upper)

        # Output only tape with black background
        output = cv2.bitwise_and(image2, image2, mask=mask)

        return output

    ##
    # @brief Draws the bounding boxes onto a frame
    # @param picks List of bounding boxes
    # @param frame Frame to draw the bounding boxes on
    # @param color RGB tupple representing the color of the bounding boxes
    # @param drawName Whether or not to add the name to the box
    # @param name The name to be displayed
    # @return boxedFrame A copy of frame with the bounding boxes drawn
    def drawBoxes(self,
                  picks,
                  frame,
                  color=(0, 255, 0),
                  drawName=False,
                  name=None):
        boxedFrame = frame.copy()
        for (xA, yA, xB, yB) in picks:
            cv2.rectangle(boxedFrame, (xA, yA), (xB, yB), color, 2)
            if drawName:
                cv2.putText(boxedFrame,
                            name,
                            # self.classifier.getClassNames[boxClass],
                            (xA, yA),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            .8,
                            color,
                            2)

        return boxedFrame

    ##
    # @brief Normalizes an image via equalization of histograms
    # @param frame The image to be normalized
    # @return frame The normalized frame
    def normalizeFrame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(frame)
        channels[0] = cv2.equalizeHist(channels[0])
        frame = cv2.merge(channels)
        frame = cv2.cvtColor(frame, cv2.COLOR_YCR_CB2BGR)
        return frame

    ##
    # @brief Finds the corners of a mask
    # @param mask Mask of the image you want to find corners of
    # @pre The mask must be black and white, and only 4 sided polygons will work
    # @return rect Array containing the four corners
    # @return imgCnt The contour of mask used to find the corners
    def getCorners(self, mask):
        (_, cnts, _) = cv2.findContours(mask.copy(),
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
        # loop over our contours
        imgCnt = None
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx) == 4:
                imgCnt = approx
                break

        # Find the corners of the contour
        pts = imgCnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        # the top-left point has the smallest sum whereas the
        # bottom-right has the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # compute the difference between the points -- the top-right
        # will have the minumum difference and the bottom-left will
        # have the maximum difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # print "rect: ", rect
        # cv2.rectangle(thing,
        # (rect[0][0], rect[0][1]),
        # (rect[2][0], rect[2][1]),
        # (100, 100, 100),
        # 2)

        return rect, imgCnt

    ##
    # @brief Uses ray tracing to determine if a point is in a polygon
    # http://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-
    #       2d-point-is-within-a-polygon
    # @param polyPoint List of points of the corners of the polygon
    # @param testPoint The test point that we are checking
    # @return Boolean representing if the point is within the polygon
    def inPolygon(self, polyPoints, testPoint):
        vertx = [polyPoint[0] for polyPoint in polyPoints]
        verty = [polyPoint[1] for polyPoint in polyPoints]
        testx = testPoint[0]
        testy = testPoint[1]

        j = len(vertx) - 1
        inP = False
        for i in range(0, len(vertx)):
            junk = (verty[i] > testy != verty[j] > testy)
            otherjunk = ((vertx[j]-vertx[i]) * (testy-verty[i]) /
                         (verty[j]-verty[i]) + vertx[i])
            if (junk & (testx < otherjunk)):
                inP = not inP
            j = i

        return inP