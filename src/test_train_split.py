import os
from natsort import natsort_keygen, ns
import shutil
cwd=os.getcwd()
if not os.path.isdir('dataset_split'):
    os.mkdir('dataset_split')
    dataset_split_path = cwd + '/dataset_split/'
    os.mkdir(dataset_split_path + '/train')
    os.mkdir(dataset_split_path + '/test')
dataset_split_path = cwd + '/dataset_split/'
original_dataset_path = cwd+'/dataset/'


face_codes_dir = os.listdir(original_dataset_path)
for face_codes in face_codes_dir:
    face = os.listdir(original_dataset_path + str(face_codes))
    natsort_key1 = natsort_keygen(key=lambda y: y.lower())
    face.sort(key=natsort_key1)
    name_code_folder = face[0].split('_')[0]

    count=0
    for face_image in face:
        if count < 3:
            if not os.path.isdir(dataset_split_path + 'train/'+str(name_code_folder)):
                os.mkdir(dataset_split_path +'train/'+ str(name_code_folder))
            shutil.copy("dataset/" + str(name_code_folder) + '/' + str(face_image),"dataset_split/"+'train/' + str(name_code_folder) + '/' + str(face_image))
            # os.rename("dataset/"+str(name_code_folder)+'/'+str(face_image), "dataset_split/"+'train/'+str(name_code_folder)+'/'+str(face_image))

            count=count+1
        else:
            if not os.path.isdir(dataset_split_path + 'test/'+str(name_code_folder)):
                os.mkdir(dataset_split_path +'test/'+str(name_code_folder))
            # os.rename("dataset/" +str(name_code_folder) + '/' + str(face_image),"dataset_split/" +'test/'+ str(name_code_folder) + '/' + str(face_image))
            shutil.copy("dataset/"  +str(name_code_folder) + '/' + str(face_image),"dataset_split/"+'test/' + str(name_code_folder) + '/' + str(face_image))
