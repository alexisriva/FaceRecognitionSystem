#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import face_recognition
import numpy as np


# In[2]:


data_path = 'data/'
test_path = 'test_data/'
outpath = 'output/'

# 'id':'name'
photo_dic = {}

# Array of known face encodings 
known_face_encodings = []
known_face_ids=[]

test_data = []


# In[3]:


for dirname, dirnames, filenames in os.walk(data_path):
    for filename in filenames:
        photo = os.path.join(dirname, filename)
#         print(photo)
        photo_data = photo.split("/")[2].split(".")[0].split("_")
        photo_id = photo_data[1]
        photo_name = "{} {}".format(photo_data[3], photo_data[4])
        if photo_id not in photo_dic:
            photo_dic[photo_id] = photo_name

print(photo_dic)
print(photo_dic[photo_id])


# In[4]:


# recognizer = cv2.face.LBPHFaceRecognizer_create()
# detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# In[4]:


# face_encoding - the model learns how to recognize it
# face_id - to use it on the dictionary
# photo_type - is grupal or not.

def getImagesAndLabels(data_path): 
    total_test = 0
    total_data = 0
    encoded_at_first_try = 0
    encoded_with_locations = 0
    no_encoded = 0
    
    for dirname, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            photo_path = os.path.join(dirname, filename)
            #print(photo_path)
            face_id =int(photo_path.split("/")[2].split(".")[0].split("_")[1])
            photo_type = photo_path.split("/")[2].split(".")[0].split("_")[-1]
            
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
            
            print(str(no_encoded) +" de "+str(total_data)+" sin encoding")
            print(str(total_test) + " para test")
    return known_face_encodings, known_face_ids


# In[5]:


def getImagesToTest(test_path):

    for dirname, dirnames, filenames in os.walk(test_path):
        for filename in filenames:
            photo_path = os.path.join(dirname, filename)
            print(photo_path)
            test_data.append(photo_path)
                       
    return test_data


# In[6]:


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


# In[7]:


known_face_encodings, known_face_ids = getImagesAndLabels(data_path)


# In[8]:


test_data = []
test_data = getImagesToTest(test_path)
for unknown_image in test_data:
    #face_locations = face_recognition.face_locations(rgb_image)
    image_loaded = face_recognition.load_image_file(unknown_image)
    rgb_image = image_loaded[...,::-1]
    face_locations = face_recognition.face_locations(rgb_image)
    #face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    #print(face_encodings)
#     Id=""
    scaledFont = 1.0 * scaleFont(unknown_image)
    face_ids=[]
    print(len(face_encodings))
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        Id = -1
    
    #print(unknown_image)
    #print(matches)
       
        if True in matches:
            first_match_index = matches.index(True)
            Id = known_face_ids[first_match_index]
            face_ids.append(Id)

    for (top, right, bottom, left), Id in zip(face_locations, face_ids):
        
        # Draw a box around the face
        rec_image = cv2.rectangle(rgb_image, (left, top), (right, bottom), (0, 0, 255), int(3*scaledFont))
        
        # Draw a label with a name below the face
#         cv2.rectangle(image_loaded, (left, bottom - (35*scaledFont)), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        imagen = cv2.putText(rec_image, photo_dic.get(str(Id), "No hubo match"), (left + 6, bottom + int(30*scaledFont)), font, scaledFont, (0, 0, 255), int(2*scaledFont))
        output = "{}img_{}.jpg".format(outpath, Id)
        cv2.imwrite(output, imagen)
        
    # Display the resulting image
    #cv2.imshow('Video', imagen)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows() 
    print(face_ids)
print(test_data)


# In[9]:


print(test_data)
print(known_face_ids)
'''
faces,Ids = getImagesAndLabels(data_path)
recognizer.train(faces, np.array(Ids))
#recognizer.save('trainner/trainner.yml')
'''


# In[13]:


'''
# We point OpenCV's CascadeClassifier function to where our 
# classifier (XML file format) is stored
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

# Load our image then convert it to grayscale
"""for Id,photo in test_data:
    image = cv2.imread(photo)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = recognizer.predict(gray)
    
    print(Id == results[0])"""

image = cv2.imread('data/h_andrade/alarconyepezallansamuel_79578_1302986_Alarcón_Allan_frontal.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray, 1.3, 5)

if not faces:
    return image, []

for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
    roi = image[y:y+h, x:x+w]
    roi = cv2.resize(roi, (200, 200))

results = recognizer.predict(gray)    

cv2.putText(image, photo_dic[str(results[0])], (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
cv2.imshow('Face Recognition', image )

cv2.waitKey(0)
cv2.destroyAllWindows()
'''


# In[5]:




