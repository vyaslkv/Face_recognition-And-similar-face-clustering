import numpy as np
import time
import cv2
import os
cam = cv2.VideoCapture(0)
# cam = cv2.VideoCapture('output1_employees_catwalk.avi')
# cam.open("rtsp://admin:99%24inmedi@192.168.0.3:554/ISAPI/streaming/channels/101")

#w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cam.set(cv2.CAP_PROP_FOCUS, 0)
#cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
#cam.set(cv2.CAP_PROP_EXPOSURE, 0.03)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
cam.set(cv2.CAP_PROP_FPS, 10)
# cam.set(cv2.CAP_PROP_EXPOSURE, 0.0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cwd = os.getcwd()
# path = cwd+'/static'
path1 = cwd +'/saved_images/'
# print (w,h)
img_counter = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output1_employees_catwalkdup.avi',fourcc, 10,(1280,720))
print(cam.isOpened())
start_time=time.time()
video_counter=0
out = cv2.VideoWriter('output1_employees_automated'+str(video_counter)+'.avi', fourcc, 10, (1280, 720))
while cam.isOpened():
    print("reading the frame")
    ret, frame = cam.read()
    print(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # out = cv2.VideoWriter('output1_employees_catwalkdup' + str(0) + '.avi', fourcc, 10, (1280, 720))

    out.write(frame)

    print(cam.get(cv2.CAP_PROP_FPS))
    if not ret:
        break
    cv2.imshow("frame",frame)
    key = cv2.waitKey(50)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'): # wait for 's' key to save and exit
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(os.path.join(path1, img_name), frame)
        print("{} written!".format(img_name))
        img_counter += 1
    end_time = time.time()
    time_cal = (end_time - start_time)/60
    print("time diff", time_cal)
    # time_cal=(end_time-start_time)/60
    if time_cal>=60:
        start_time=0
        video_counter +=1
        out = cv2.VideoWriter('output1_employees_automated' + str(video_counter) + '.avi', fourcc, 10, (1280, 720))



cam.release()
out.release()
cv2.destroyAllWindows()

# When everything done, release the capture
