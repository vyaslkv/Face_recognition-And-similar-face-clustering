#!/usr/bin/python
import dlib
from imutils import paths
import more_itertools
import cv2
import time
# cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
detector = dlib.get_frontal_face_detector()
win = dlib.image_window()
# imagePaths = list(paths.list_images("saved_images"))
cam = cv2.VideoCapture('output1_employees_catwalk4_edited_manish_ji.mp4')
# for f in imagePaths:
face_counter=0
start_time=time.time()
frame_counter=0
while True:
    ret, f = cam.read()
    if not ret:
        break

    # print("Processing file: {}".format(f))
    # img = dlib.load_rgb_image(f)
    img=f
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(img, 1)
    # dets = cnn_face_detector(img, 1)
    # print("ddddddddddddeeeeeeeeeeeeeeeeet",dets)

    if dets:
        for d in dets:
            # print(type(d))
            # print("dddddddddddddddddddddddd",list(more_itertools.collapse(dets)))
            left = d.left()
            top = d.top()
            right = d.right()
            bottom = d.bottom()


            cropped_face = img[top-50:bottom+50, left-50:right+50]
            cv2.imwrite("saved_images/Cropped_Faces"+str(face_counter)+".png",cropped_face)
            face_counter += 1
    print("Number of faces detected: {}".format(len(dets)))

    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))

    # win.clear_overlay()
    # win.set_image(img)
    # win.add_overlay(dets)
    # cv2.imshow("frame",f)
    frame_counter +=1
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    # dlib.hit_enter_to_continue()
end_time=time.time()
print("total number of frames processed",frame_counter)
print("total time taken",start_time-end_time)
cam.release()
cv2.destroyAllWindows()



# Finally, if you really want to you can ask the detector to tell you the score
# for each detection.  The score is bigger for more confident detections.
# The third argument to run is an optional adjustment to the detection threshold,
# where a negative value will return more detections and a positive value fewer.
# Also, the idx tells you which of the face sub-detectors matched.  This can be
# used to broadly identify faces in different orientations.
# if (len(imagePaths) > 0):
#     img = dlib.load_rgb_image(sys.argv[1])
#     dets, scores, idx = detector.run(img, 1, -1)
#     for i, d in enumerate(dets):
#         print("Detection {}, score: {}, face_type:{}".format(
#             d, scores[i], idx[i]))
