#!/usr/bin/env python
# coding: utf-8
import os
import cv2
import face_recognition
import numpy as np
import pickle

data_path = 'data/'
test_path = 'test_data/'
outpath = 'output/'

def createDic(data_path=data_path):
    # 'id':'name'
    photo_dic = {}

    for dirname, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            photo = os.path.join(dirname, filename)
            # print(photo)
            photo_data = photo.split("/")[-1].split(".")[0].split("_")
            photo_id = photo_data[1]
            photo_name = "{} {}".format(photo_data[3], photo_data[4])
            if photo_id not in photo_dic:
                photo_dic[photo_id] = photo_name
    
    return photo_dic

# print(photo_dic)
# print(photo_dic[photo_id])


def getImagesAndLabels(data_path=data_path):
    dic = createDic(data_path)
    total_test = 0
    total_data = 0
    encoded_at_first_try = 0
    encoded_with_locations = 0
    no_encoded = 0
    test_data = []
    # Array of known face encodings 
    known_face_encodings = []
    known_face_ids=[]
    all_face_encodings = {}
    
    for dirname, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            photo_path = os.path.join(dirname, filename)
            #print(photo_path)
            face_id = int(photo_path.split("/")[-1].split(".")[0].split("_")[1])
            photo_type = photo_path.split("/")[-1].split(".")[0].split("_")[-1]
            
            if photo_type.lower() == "grupal":
                test_data.append(photo_path)
                total_test += 1
            else:
                total_data += 1
                image_loaded = face_recognition.load_image_file(photo_path)
                face_encoding = face_recognition.face_encodings(image_loaded)

                if face_encoding == []:
                    img = cv2.imread(photo_path,0)
                    height, width = img.shape[:2]
                    known_face_locations=[(0, width, height, 0)]
                    face_encoding = face_recognition.face_encodings(image_loaded,known_face_locations)

                    if face_encoding == []:
                        no_encoded += 1
                        continue
                    else:
                        encoded_with_locations += 1
                else:
                    encoded_at_first_try += 1

                face_encoding = face_encoding[0]
                
                known_face_encodings.append(face_encoding)
                known_face_ids.append(face_id)
                # if dic[str(face_id)] in all_face_encodings:
                #     all_face_encodings[dic[str(face_id)]].append(face_encoding)
                # else:
                #     all_face_encodings[dic[str(face_id)]] = [face_encoding]
            
            print(str(no_encoded) +" de "+str(total_data)+" sin encoding")
            print(str(total_test) + " para test")
    print("Entrenado")
    # with open('test.txt', 'ab') as outfile:
    #     pickle.dump(all_face_encodings, outfile)
    return known_face_encodings, known_face_ids, dic


def getImagesToTest(test_path=test_path):

    test_data = []

    for dirname, dirnames, filenames in os.walk(test_path):
        for filename in filenames:
            photo_path = os.path.join(dirname, filename)
            print(photo_path)
            test_data.append(photo_path)
                       
    return test_data


def scaleFont(photo):
    img = cv2.imread(photo,0)
    height, width = img.shape[:2]
    min_value = min(height,width)
    max_value = max(height,width)
    if min_value < 900:
        return min_value/900
    elif max_value > 1300:
        return max_value/1300
    return 1

# known_face_encodings, known_face_ids = getImagesAndLabels(data_path)

def recognizePeople(known_face_encodings, known_face_ids, photo_dic, test_path=test_path, outpath=outpath):
    # test_data = []
    # print(known_face_encodings)
    # print(known_face_ids)
    test_data = getImagesToTest(test_path)
    # all_face_encodings = {}
    # with open('test.txt', 'rb') as infile:
    #     all_face_encodings = pickle.load(infile)
    # print(all_face_encodings)
    # known_face_encodings = np.array(list(all_face_encodings.values()))
    # print(known_face_encodings_from_file[0] == known_face_encodings[0])
    # known_face_names = list(all_face_encodings.keys())
    for unknown_image in test_data:
        image_loaded = face_recognition.load_image_file(unknown_image)
        rgb_image = image_loaded[...,::-1]
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        scaledFont = 1.0 * scaleFont(unknown_image)
        face_ids=[]
        # face_names = []
        print(len(face_encodings))
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            Id = -1
            # name = "Unknown"
        
            if True in matches:
                first_match_index = matches.index(True)
                Id = known_face_ids[first_match_index]
                # name = known_face_names[first_match_index]
                face_ids.append(Id)
                # face_names.append(name)

        for (top, right, bottom, left), Id in zip(face_locations, face_ids):
            # Draw a box around the face
            rec_image = cv2.rectangle(rgb_image, (left, top), (right, bottom), (0, 0, 255), int(3*scaledFont))
            
            # Draw a label with a name below the face
            font = cv2.FONT_HERSHEY_DUPLEX
            imagen = cv2.putText(rec_image, photo_dic.get(str(Id)), (left + 6, bottom + int(30*scaledFont)), font, scaledFont, (0, 0, 255), int(2*scaledFont))
            output = "{}/img_{}.jpg".format(outpath, Id)
            cv2.imwrite(output, imagen)
        
        # Display the resulting image
        #cv2.imshow('Video', imagen)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows() 
        print(face_ids)
    print(test_data)
