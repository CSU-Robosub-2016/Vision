'''
camera.py - Extend this class through inheritance when adding different
cameras.  If the camera needs calibration during initialization, then
add that to the __init__ function inside the extended camera class.
For compliance with the framework, the extended camera class MUST
implement at least the getFrame function and call the parent __init__
function during its __init__ call
'''


class Camera:
    ##
    # @param connType The connection type to the camera.  0=HDMI, 1=USB
    # @param name Name of the camera
    # @param debug Used for debug purposes
    # @pre The IP camera must be alive, connected and accepting requests
    # @post An openCV connection to the camera
    def __init__(self, connType=99, name="default", debug=None):
        self.connType = connType
        # Decode of the connection type
        if self.connType == 0:
            self.connTypeStr = "HDMI"
        elif self.connType == 1:
            self.connTypeStr = "USB"
        elif self.connType == 99:
            self.connTypeStr = "DEBUG"
        else:
            self.connTypeStr = 'DEMONS!!!'

        self.name = name
        self.debug = debug

    def getName(self):
        return self.name

    def setName(self, name):
        self.name = name

    def getconnType(self):
        return self.connType

    def setconnType(self, connType):
        self.connType = connType

    def getconnTypeStr(self):
        return self.connTypeStr

    def setconnTypeStr(self, connTypeStr):
        self.connTypeStr = connTypeStr

    def __str__(self):
        if self.debug is not None:
            return "Camera name: %s, Connection type: %s, Debug: %s" % (
                self.name, self.connTypeStr, self.debug)

        return "Camera name: %s, Connection type: %s" % (
            self.name, self.connTypeStr)
