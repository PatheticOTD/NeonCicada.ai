import os
import cv2
import sys
import pickle
import numpy as np
import face_recognition


area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 1] - x[:, 3])


class FaceModel():
    def __init__(self, encodings_path = None, labels_path = None):
        self.encodings = None
        self.labels = None
        
        if (encodings_path != None) and (labels_path != None):
            with open(encodings_path, 'rb') as f:
                self.encodings = pickle.load(f)
            with open(labels_path, 'rb') as f:
                self.labels = pickle.load(f)

    def fit(self, data_path):
        images = []
        self.labels = []
        self.encodings = []

        area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 1] - x[:, 3])
        for folder in os.listdir(data_path):
            folder_path = os.path.join(data_path, folder)
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path,  filename)
                image = face_recognition.load_image_file(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                coords = np.array(face_recognition.face_locations(image))
                if len(coords) > 0:
                    coords = coords[np.argmax(area(coords))]
                    image = image[coords[0]:coords[2], coords[3]: coords[2]]
                    image = cv2.resize(image,(160, 160))
                    images.append(image)
                    labels.append(folder)       
        images = np.array(images)
        labels = np.array(labels)
        for i in images:
            self.encodings.append(face_recognition.face_encodings(i, known_face_locations=[(0, 160, 160, 0)])[0])

        with open('data/train_imgs.pkl', 'wb') as f:
            pickle.dump(images, f)
        with open('data/train_labels.pkl', 'wb') as f:
            pickle.dump(labels, f)
        with open('data/encodings.pkl', 'wb') as f:
            pickle.dump(self.encodings, f)
        
        return 0
            
    def predict(self, img_path = 'data/validate/Hugh Jackman/Hugh Jackman17293.jpg'):
        
        image = face_recognition.load_image_file(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        coords = np.array(face_recognition.face_locations(image))
        coords = coords[np.argmax(area(coords))]
        
        image = image[coords[0]:coords[2], coords[3]: coords[1]]
        image = cv2.resize(image,(160, 160))
        
        encoding = face_recognition.face_encodings(image, known_face_locations=[(0, 160, 160, 0)])[0]
        filter = face_recognition.compare_faces(self.encodings, encoding, tolerance=0.4)
        
        if sum(filter) > 0:
            return self.labels[filter]
        else: return ['Unknown']