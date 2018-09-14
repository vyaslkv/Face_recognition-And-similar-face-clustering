import face_recognition
import cv2
import glob
import numpy as np

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

persons_encodings={}
# Load a sample picture and learn how to recognize it.
for img in glob.glob("./data/*.jpg"):
    #print(img)
    person_image = face_recognition.load_image_file(img)
    person_face_encoding = face_recognition.face_encodings(person_image)[0]
    persons_encodings[img.split("/")[2].split("_")[0]] = person_face_encoding# use '_' for split in new database

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)#,model='cnn')
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(list(persons_encodings.values()), face_encoding,tolerance=0.6)

            # See how far apart the test image is from the known faces
            face_distances = face_recognition.face_distance(list(persons_encodings.values()), face_encoding)
            print(list(persons_encodings.keys()))
            print(face_distances)
            #for i, face_distance in enumerate(face_distances):
            #    print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
            #    print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
            #    print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(face_distance < 0.5))
            #    print()

            name = "Unknown"
            matches  = [i for i, x in enumerate(match) if x]
            if matches:
                name = list(persons_encodings.keys())[np.argmin(face_distances)]


            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
