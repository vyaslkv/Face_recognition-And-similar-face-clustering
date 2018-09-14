from imutils import paths
# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png

# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
# ap.add_argument("-e", "--encodings", required=True,
# 	help="path to serialized db of facial encodings")
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
pkl=[10,40,80,120]
for i in pkl:
    print("[INFO] loading encodings...")
    data = pickle.loads(open("encodings"+str(i)+'.pickle', "rb").read())
    imagePaths = list(paths.list_images(args["dataset"]))
    print("lennnnnnnnnnnnnnnnnnnnn",len(imagePaths))
    count=0
    for (j,image) in enumerate(imagePaths):
        j,image= (j,image)
        if j<i:
            continue
        name_id= image.split('/')[1]
    # load the input image and convert it from BGR to RGB
        image = cv2.imread(image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes corresponding
        # to each face in the input image, then compute the facial embeddings
        # for each face
        print("[INFO] recognizing faces...")
        boxes = face_recognition.face_locations(rgb,
                                                model=args["detection_method"])
        encodings = face_recognition.face_encodings(rgb, boxes)

        # initialize the list of names for each face detected
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # print("encodingsssssssss", encoding)
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding,tolerance=0.4)
            name = "Unknown"
            # print("matches",matches)
            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number of
                # votes (note: in the event of an unlikely tie Python will
                # select first entry in the dictionary)
                print("count dict",counts)
                name = max(counts, key=counts.get)
                print("namee id",name_id)
                if name==name_id:
                    count=count+1
                    print(count)
            # update the list of names

            names.append(name)
        print("names lis",names)
    print("count"+str(i)+"changeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",count)
        # loop over the recognized faces
        # for ((top, right, bottom, left), name) in zip(boxes, names):
        #     # draw the predicted face name on the image
        #     cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        #     y = top - 15 if top - 15 > 15 else top + 15
        #     cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.75, (0, 255, 0), 2)

        # show the output image
        # cv2.imshow("Image", image)
        # cv2.waitKey(0)