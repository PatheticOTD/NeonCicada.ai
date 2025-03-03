import os
import cv2
import sys
import pickle
import numpy as np
import face_recognition
from ultralytics import YOLO
from insightface.app import FaceAnalysis

area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 1] - x[:, 3])


class FaceNetModel():
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
        
class ArcFaceModel():
    def __init__(self, encodings_path = None, labels_path = None, img_size=256, yolo_model_path = 'models/model.pt'):
        self.encodings = None
        self.labels = None
        
        if (encodings_path != None) and (labels_path != None):
            with open(encodings_path, 'rb') as f:
                self.encodings = pickle.load(f)
            with open(labels_path, 'rb') as f:
                self.labels = pickle.load(f)
        self.img_size = img_size
        self.model = YOLO(yolo_model_path)
        self.app = FaceAnalysis(name='buffalo_l', 
                   allowed_modules=['detection', 'recognition'], 
                   providers=['CPUExecutionProvider'], 
                   det_thresh=0.5)  # Use 'CUDAExecutionProvider' for GPU
        self.app.prepare(ctx_id=-1, det_size=(img_size, img_size))
        
    def fit(self, data_path):
        images = []
        self.labels = []
        self.encodings = []
        for folder in os.listdir(data_path):
            folder_path = os.path.join(data_path, folder)
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                image = cv2.imread(img_path)
                
                if image is None:  # Пропускаем битые файлы
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.model.predict(image, conf=0.7)
                
                max_area = 0
                closest_face = None
                
                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = box.astype(int)  # Преобразуем сразу все координаты
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            closest_face = (x1, y1, x2, y2)

                if closest_face is not None:
                    x1, y1, x2, y2 = closest_face
                    
                    # Добавляем проверку границ и размера области
                    h, w = image.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    # Проверяем валидность региона
                    if x2 > x1 and y2 > y1:
                        face_roi = image[y1:y2, x1:x2]  # Правильный порядок среза [y, x]
                        
                        if face_roi.size > 0:  # Проверка на пустую область
                            resized = cv2.resize(face_roi, (self.img_size, self.img_size))
                            images.append(resized)
                            self.labels.append(folder)
        for i in images:
            face = self.app.get(i)
            self.encodings.append(face[0].embedding)
                
        images = np.array(images)
        self.labels = np.array(self.labels)
        self.encodings = np.array(self.encodings)
        
        with open('data/train_imgs(arcface).pkl', 'wb') as f:
            pickle.dump(images, f)
        with open('data/train_labels(arcface).pkl', 'wb') as f:
            pickle.dump(self.labels, f)
        with open('data/encodings(arcface).pkl', 'wb') as f:
            pickle.dump(self.encodings, f)
                    
    def compare_faces(self, query_emb, threshold=0.65):
        """
        Сравнивает эмбеддинг с массивом эмбеддингов и возвращает индекс лучшего совпадения
        
        Параметры:
        query_emb (np.array): Эмбеддинг для сравнения
        emb_list (list[np.array]): Список эталонных эмбеддингов
        threshold (float): Порог для определения совпадения
        Рекомендации:

    Для идентификации: VGGFace2, Glint360K или CASIA-WebFace (баланс размера и качества).

    Для атрибутов: CelebA или FairFace.

    Для тестирования: LFW.
        Возвращает:
        tuple: (индекс лучшего совпадения, косинусная схожесть) или (-1, 0) если совпадений нет
        """
        if type(self.encodings) != np.ndarray:
            return (-1, 0.0)
        
        # Преобразуем в numpy array для векторизованных вычислений
        emb_array = np.array(self.encodings)
        
        # Вычисляем косинусную схожесть
        similarities = np.dot(emb_array, query_emb) / (
            np.linalg.norm(emb_array, axis=1) * np.linalg.norm(query_emb)
        )
        
        # Находим индекс максимальной схожести
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        # Проверяем порог
        if best_similarity > threshold:
            return (best_match_idx, best_similarity)
        else:
            return (-1, best_similarity)
    
    def predict(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model.predict(image, conf=0.7)
        
        max_area = 0
        closest_face = None
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)  # Преобразуем сразу все координаты
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    closest_face = (x1, y1, x2, y2)

        if closest_face is not None:
            x1, y1, x2, y2 = closest_face
            
            # Добавляем проверку границ и размера области
            h, w = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, (self.img_size, self.img_size))
        encoding = self.app.get(face)[0].embedding
        
        filter = self.compare_faces(encoding, threshold=.9)
        
        if filter[0] != -1:
            return [self.labels[filter[0]]]
        else: return filter
    
if __name__ == '__main__':
    test = ArcFaceModel()
    test.fit('data/train')
    print(test.predict('data/validate/Павел Михно/Mihno.jpg'))