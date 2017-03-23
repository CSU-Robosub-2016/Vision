import numpy as np
import cv2

#Last edited by Brett on 11/27/16

#Read in original video
im = cv2.VideoCapture('C:/Users/Brett/Pictures/RoboSub/Test1.MP4')

# Define the codec and create VideoWriter object

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('C:/Users/Brett/Pictures/output2.avi',fourcc, 60.0, (848,480), False)

while(im.isOpened()):
    ret, frame = im.read()
    if ret==True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Blur to remove noise
        im2 = cv2.medianBlur(frame,5)

    #Threshold filter to find contours
        ret,thresh = cv2.threshold(im2, 120, 255 ,0)
    #Find Contours
        q, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]

    #Create Rectangle around largest contour (ROI)
        x,y,w,h = cv2.boundingRect(cnt)
        new= cv2.rectangle(im2,(x,y),(x+w,y+h),(0,255,0),2)

    #Compare region of interest to original image, set everything outside the ROI to zero
        mask = np.zeros(new.shape,np.uint8)
        mask[y:y+h,x:x+w] = new[y:y+h,x:x+w]
        out2 = cv2.bitwise_and(new, mask)

    #Set upper and lower limits for colors
        lower = 40
        upper = 255

    #Filter out all colors except the tape
        mask = cv2.inRange(out2, lower, upper)

    #Output only tape with black background
        output = cv2.bitwise_and(out2, out2, mask = mask)

        out.write(output)

    #Show output compared to original

        cv2.imshow("Original with ROI", im2)
        cv2.imshow("Final", output)
        cv2.waitKey(1) #change value to make video faster
    else:
        break
im.release()
out.release()
cv2.destroyAllWindows()
