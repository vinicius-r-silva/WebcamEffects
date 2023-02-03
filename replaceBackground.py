# sudo modprobe -r v4l2loopback
# sudo modprobe v4l2loopback devices=1 video_nr=20 card_label="fakeWebcam" exclusive_caps=1
# v4l2-ctl --list-devices
# ffplay /dev/video20
# watch -n 0.5 "lsmod | grep v4l2loopback"
# watch -n 0.5 nvidia-smi

import os
import cv2
import time
import pyfakewebcam
from webcamHelper import WebcamVideoStream
import subprocess

def isWebcamBusy():
    module_name = 'v4l2loopback'
    lsmod_proc = subprocess.Popen(['lsmod'], stdout=subprocess.PIPE)
    grep_proc = subprocess.Popen(['grep', module_name], stdin=lsmod_proc.stdout, stdout=subprocess.PIPE)
    sRes = grep_proc.communicate()[0].splitlines()
    bFirstline = sRes[0]
    sFirstline = bFirstline.decode()
    val = int(sFirstline[len(sFirstline) - 1])
    return val == 2

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

back_visum = cv2.imread("fundoMeeting.png")
dim = (output_res['w'], output_res['h'])
back_visum = cv2.resize(back_visum, dim, interpolation = cv2.INTER_AREA)
freezeFrame = cv2.cvtColor(back_visum, cv2.COLOR_BGR2RGB)


#setup the neural network
from rvmHelper import rvm
filters = rvm(input_res, output_res, back_visum)

webcamBusy = True
steps = 1
interval = 0.030
start = time.time()
print("loop init")
while(True):
    steps += 1
    if(steps == 20000):
        steps = 1
        start = time.time()


    if (webcamBusy and steps % 50 == 0):
        webcamBusy = isWebcamBusy()

    if(not webcamBusy):
        print('free memory')
        del filters
        filters = None
        vs.stop()
        del vs

    while(not webcamBusy):
        time.sleep(1)
        filters = None
        fake.schedule_frame(freezeFrame)
        webcamBusy = isWebcamBusy()


    if(filters == None):
        print('init memory')
        filters = rvm(input_res, output_res, back_visum)
        vs = WebcamVideoStream(input_res = input_res, src = '/dev/video0').start()
        time.sleep(1.0)
        steps = 1
        start = time.time()
    else:
        while time.time() < (start + interval * steps):
            time.sleep(0.01)
            pass

    frame = vs.read()
    # orig = frame.copy()
    res = filters.replaceBackground(frame)

    # cv2.imshow("Original", orig)
    # cv2.imshow("Filter", res)
    # k = cv2.waitKey(1) & 0xff
    # if k == 27:
    #     break

    fake.schedule_frame(res)
