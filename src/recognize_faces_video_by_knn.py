# USAGE
# python recognize_faces_video.py --encodings encodings.pickle
# python recognize_faces_video.py --encodings encodings.pickle --output output/jurassic_park_trailer_output.avi --display 0

# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-e", "--encodings", required=True,
# 	help="path to serialized db of facial encodings")

ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
# print("[INFO] loading encodings...")
# data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
# source= input()

# vs = VideoStream(src='"rtsp://admin:99%24inmedi@192.168.0.3:554/ISAPI/streaming/channels/101"').start()

# vs = VideoStream(src=1).start()
# vs.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
# vs.set(cv2.CAP_PROP_FPS, 60)
cam = cv2.VideoCapture('output1_employees_catwalk_knn_testing.avi')
# cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
# cam.set(cv2.CAP_PROP_FOCUS, 0)
# cam.set(cv2.CAP_PROP_FPS, 10)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cam.open("rtsp://admin:99%24inmedi@192.168.0.3:554/ISAPI/streaming/channels/101")
writer = None
import os
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# time.sleep(2.0)
def predict(X_img, knn_clf=None, model_path=None, distance_threshold=0.4):

    # if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
    #     raise Exception("Invalid image path: {}".format(X_img_path))
	#
    # if knn_clf is None and model_path is None:
    #     raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    # X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img,model='hog')

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)

    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	names = []
	cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('H','2','6','4'))
	ret,frame = cam.read()
	print(cam.get(cv2.CAP_PROP_FPS))
	if frame is not None:
		# print(len(frame[0]))
		# print(type(frame[0]))
		# print(len(frame[1]))
		# cv2.imshow("dfdf",frame)
		# cv2.waitKey(0)
		# print(vs.get(cv2.CAP_PROP_FPS))
		rgb = imutils.resize(frame, width=750)
		r = frame.shape[1] / float(rgb.shape[1])
		predictions = predict(rgb, model_path='knn_trained_no_front_5people')
		# print(predictions[0][0])
		# print(type(predictions[0][0]))
		# name = predictions[0][0]
		# convert the input frame from BGR to RGB then resize it to have
		# a width of 750px (to speedup processing)
		# rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# rgb = imutils.resize(frame, width=750)
		# r = frame.shape[1] / float(rgb.shape[1])
		#
		# # detect the (x, y)-coordinates of the bounding boxes
		# # corresponding to each face in the input frame, then compute
		# # the facial embeddings for each face
		# boxes = face_recognition.face_locations(rgb,
		# 	model=args["detection_method"])
		# encodings = face_recognition.face_encodings(rgb, boxes)
		# names = []
		#
		# # loop over the facial embeddings
		# for encoding in encodings:
		# 	# attempt to match each face in the input image to our known
		# 	# encodings
		# 	matches = face_recognition.compare_faces(data["encodings"],
		# 		encoding)
		# 	name = "Unknown"
		#
		# 	# check to see if we have found a match
		# 	if True in matches:
		# 		# find the indexes of all matched faces then initialize a
		# 		# dictionary to count the total number of times each face
		# 		# was matched
		# 		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		# 		counts = {}
		#
		# 		# loop over the matched indexes and maintain a count for
		# 		# each recognized face face
		# 		for i in matchedIdxs:
		# 			name = data["names"][i]
		# 			counts[name] = counts.get(name, 0) + 1
		#
		# 		# determine the recognized face with the largest number
		# 		# of votes (note: in the event of an unlikely tie Python
		# 		# will select first entry in the dictionary)
		# 		name = max(counts, key=counts.get)

			# update the list of names
		# names.append(name)

		# loop over the recognized faces
		# print(predictions[0])
		for (name,(top, right, bottom, left)) in predictions:
			# rescale the face coordinates
			# print(type(r))
			# print(type(top))
			top = int(int(top) * r)
			right = int(int(right) * r)
			bottom = int(int(bottom) * r)
			left = int(int(left) * r)

			# draw the predicted face name on the image
			cv2.rectangle(frame, (left, top), (right, bottom),
				(0, 255, 0), 2)
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

		# if the video writer is None *AND* we are supposed to write
		# the output video to disk initialize the writer
		# if writer is None and args["output"] is not None:
		# 	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		# 	writer = cv2.VideoWriter(args["output"], fourcc, 20,
		# 		(frame.shape[1], frame.shape[0]), True)

		# if the writer is not None, write the frame with recognized
		# faces t odisk
		# if writer is not None:
		# 	writer.write(frame)

		# check to see if we are supposed to display the output frame to
		# the screen
		if args["display"] > 0:
			cv2.namedWindow("Frame",cv2.WINDOW_GUI_EXPANDED)
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break

# do a bit of cleanup
cv2.destroyAllWindows()
# cam.stop()
cam.release()
# check to see if the video writer point needs to be released
# if writer is not None:
# 	writer.release()
