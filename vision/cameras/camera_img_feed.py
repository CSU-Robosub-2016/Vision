# camera_img_feed.py - This is a 'test' camera that will take in a single image
#                      and act as a camera.
import cv2
# import imutils
from camera import Camera


class imgFeedCamera(Camera):
    ##
    # @param fileLocation The absolute file path to the image
    def __init__(self, connType=99, name="Image Feed", debug=None):
        # use the IPaddress as the video file location
        Camera.__init__(self, connType, name, debug)
        self.frame = cv2.imread(debug)
        print("Initialized camera.", self.__str__())

    ##
    # @return Processed frame that was captured from the camera
    def getFrame(self):
        return self.frame
