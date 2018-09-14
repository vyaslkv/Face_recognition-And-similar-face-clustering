import cv2
import os
cwd = os.getcwd()
path1=cwd+'/saved_images/'
def click_image():
    cam = cv2.VideoCapture('output.avi')
    cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
    cam.set(cv2.CAP_PROP_FPS, 10)
    # cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # # cam.set(cv2.CAP_PROP_FOCUS, 0.35)
    # # cam.set(cv2.CAP_PROP_FOCUS, 0.098)
    # cam.set(cv2.CAP_PROP_FOCUS, 0.30)
    #
    # cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    # # cam.set(cv2.CAP_PROP_EXPOSURE, 0.125)
    # cam.set(cv2.CAP_PROP_EXPOSURE, 0.5)
    # # cam.set(cv2.CAP_PROP_EXPOSURE, 0.0)
    # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    img_counter = 0
    while True:
        ret, frame = cam.read()
        print(cam.get(cv2.CAP_PROP_FPS))
        if not ret:
            break
        cv2.imshow("frame", frame)
        # cv2.waitKey(5)
        key = cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('s'):  # wait for 's' key to save and exit
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(os.path.join(path1, img_name), frame)
            print("{} written!".format(img_name))
            img_counter += 1
    cam.release()
    cv2.destroyAllWindows()
click_image()
