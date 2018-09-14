import cv2
import numpy as np
from time import sleep
#import libbgs
import os



cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
#cap.set(cv2.CAP_PROP_FPS, 120)
#cap.set(cv2.CAP_PROP_AUTOFOCUS,0)
#print('focus',self.video_cap.get(cv2.CAP_PROP_FOCUS))
#cap.set(cv2.CAP_PROP_FOCUS,0.08) # for logitech camera
#self.video_cap.set(cv2.CAP_PROP_BRIGHTNESS,0.4980392156862745)
#self.video_cap.set(cv2.CAP_PROP_SATURATION,0.4980392156862745)
#self.video_cap.set(cv2.CAP_PROP_GAIN,0.0)#1.0
#cap.set(cv2.CAP_PROP_EXPOSURE,0.8) # for logitech camera
# print("Exposure value :",cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
#cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,0.25)


file_name = -1
x,y,w,h = 180,50,340,350
person_name = str(input("Please enter the name of the person to be added to database:"))
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    disp_frame = np.copy(frame)
    # Draw ROI for capturing face
    cv2.rectangle(disp_frame, (x, y), (x+w, y+h), (255, 0, 0), 5)

    # Display the resulting frame
    cv2.imshow('frame', disp_frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('c'):
        file_name += 1
        fname = "./data/" +str(person_name) + '_' + str(file_name)+'.jpg' 
        cv2.imwrite(fname, frame[y:y+h,x:x+w])

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

