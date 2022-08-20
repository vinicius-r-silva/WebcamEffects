import cv2
from threading import Thread

# https://github.com/PyImageSearch/imutils/blob/master/imutils/video/webcamvideostream.py
class WebcamVideoStream:
    def __init__(self, input_res = {'h' : 1080, 'w' : 1920}, src=0, name="WebcamVideoStream"):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.codec = cv2.VideoWriter.fourcc(*"MJPG")
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FOURCC, self.codec)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, input_res['w'])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, input_res['h'])
        
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            # time.sleep(0.015)

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
