# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils     #pip install imutils on terminal
import time
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")    #video file is optional, if video file equals None, then opencv will use webcam
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")  #500 pixels, no need to process large raw images through webcam
args = vars(ap.parse_args())
# if the video argument is None, then we are reading from webcam
if args.get('video', None) is None:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
# otherwise, we are reading from a video file
else:
    vs = cv2.VideoCapture(args["video"])
# initialize the first frame in the video stream
firstFrame = None

# loop over the frames of the video
sta = 0
while True:
    frame = vs.read()
    frame = frame if args.get('video', None) is None else frame[1]
    text = 'Unoccupied'

    if  frame is None:
        break
    
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if firstFrame is None:
        firstFrame = gray
        continue
    
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]


    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    record = "No"
    for c in cnts:
        if cv2.contourArea(c) < args["min_area"]:
            record = "No"
            continue
       
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = 'Occupied'
        record = "Yes"
    
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.imshow('Motion Detector', frame)
    cv2.imshow('Thresh', thresh)
    cv2.imshow('Frame Delta', frameDelta)

    if record = "Yes":
        try:
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime("%Y.%m.%d_%H.%M")
            if sta != st:
                filename = 'video-' + st + '.mp4'
                out = cv2.VideoWriter(filename, fourcc, float(fps), (1280,720))
                frame_name = filename.replace('.mp4','.png').format(frame_index)
                cv2.imwrite(frame_name,frame)
                sta = st
            frame_record = cv2.resize(frame,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            out.write(frame_record)
        except Exception as e:
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
vs.stop()
vs.release()
cv2.destroyAllWindows()
