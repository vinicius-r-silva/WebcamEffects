# sudo modprobe -r v4l2loopback
# sudo modprobe v4l2loopback devices=1 video_nr=20 card_label="fakeWebcam" exclusive_caps=1
# v4l2-ctl --list-devices
# ffplay /dev/video20

import os
import cv2
import time
import pyfakewebcam
from webcamHelper import WebcamVideoStream

#set the outuput resolution (a few programs (Teams,..) doesnt support resolutions over 720p)
input_res = {'h' : 720, 'w' : 1280}
output_res = {'h' : 720, 'w' : 1280}
vs = WebcamVideoStream(input_res = input_res, src = '/dev/video0').start()
time.sleep(1.0)

#setup the fake webcam
os.system("sudo modprobe -r v4l2loopback")
os.system("sudo modprobe v4l2loopback devices=1 video_nr=20 card_label=\"fakeWebcam\" exclusive_caps=1")
time.sleep(1.0)

fake = pyfakewebcam.FakeWebcam('/dev/video20', output_res['w'], output_res['h'])

#setup the neural network
from rvmHelper import rvm
filters = rvm(input_res, output_res)

steps = 0
interval = 0.030
start = time.time()
print("loop init")
while(True):
    steps += 1
    if(steps == 20000):
        steps = 0
        start = time.time()

    while time.time() < (start + interval * steps):
        time.sleep(0.01)
        pass

    frame = vs.read()
    # orig = frame.copy()
    res = filters.blurBackground(frame)

    # cv2.imshow("Original", orig)
    # cv2.imshow("Filter", res)
    # k = cv2.waitKey(1) & 0xff
    # if k == 27:
    #     break

    fake.schedule_frame(res)
