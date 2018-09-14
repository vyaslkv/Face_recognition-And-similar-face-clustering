import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
import os
def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.4):

    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img,model='hog')

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    # print("closest distance",closest_distances)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


from imutils import paths


count=0
knns=[20,40,80,120]
for i in knns:
    imagePaths = list(paths.list_images('dataset'))
    print("lennnnnnnnnnn",len(imagePaths))

    for (j,image) in enumerate(imagePaths):
        j,image = (j,image)
        if j <i:
            continue
        print("jjjjjjjjjjjjjjjjjj",j)
        name_id= image.split('/')[1]
        predictions = predict(image,model_path='knn'+str(i))
        # print("predictions",predictions)
        # print(predictions[0][0])
        # print(type(predictions[0][0]))
        print("predictionsssssssssss",predictions)
        if predictions ==[]:
            continue
        name = predictions[0][0]
        # name=name.encode("UTF-8"
        if name==name_id:
            count=count+1
            print(count)
        else:
            print(name)
            print(name_id)
    print("countyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy",count)