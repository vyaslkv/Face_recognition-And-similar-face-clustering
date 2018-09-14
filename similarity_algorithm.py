import face_recognition
import cv2
import glob
import numpy as np
import os
import shutil
persons_encodings={}
# Load a sample picture and learn how to recognize it.
group_counter=0
for index1,img1 in enumerate(glob.glob("./saved_images2/*.png")):
    try:
        print("image 1", img1)
        person_image1 = face_recognition.load_image_file(img1)
        person_face_encoding1 = face_recognition.face_encodings(person_image1)[0]
    except:
        continue
    for index2,img2 in enumerate(glob.glob("./saved_images2/*.png")):

        if index2<=index1:
            continue
        print("image 2", img2)
        try:
            image = cv2.imread(img2)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            continue
        img2=img2.split('/')[2]
        print("only image 2",img2)
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb)#,model='hog')

        # compute the facial embedding for the face
        person_face_encoding2 = face_recognition.face_encodings(rgb, boxes)
        # person_image2 = face_recognition.load_image_file(img2)
        # person_face_encoding2 = face_recognition.face_encodings(person_image2)[0]
        match = face_recognition.compare_faces(person_face_encoding2,person_face_encoding1,tolerance=0.4)
        print("match result",match)
        if match==[True]:
            if not os.path.isdir('saved_images2/matched_group'+str(group_counter)):
                os.mkdir('saved_images2/matched_group'+str(group_counter))
            # if not os.path.isfile("saved_images1/saved_images1/"+ str(img2)):

            shutil.move("saved_images2/"+ str(img2), 'saved_images2/matched_group' + str(group_counter) + '/' + str(img2))
        else:
            continue
    group_counter +=1
