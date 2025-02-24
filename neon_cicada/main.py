import sys
import cv2
import numpy as np
import face_recognition
import pickle
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QTextEdit, QVBoxLayout, QFileDialog
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt
from PIL import Image, ImageDraw, ImageFont

with open('data/encodings.pkl', 'rb') as f:
    encodings = pickle.load(f)
with open('data/train_labels.pkl', 'rb') as f:
    labels = pickle.load(f)

area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 1] - x[:, 3])

def check_image(img_path = 'data/validate/Hugh Jackman/Hugh Jackman17293.jpg'):
    image = face_recognition.load_image_file(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    coords = np.array(face_recognition.face_locations(image))
    coords = coords[np.argmax(area(coords))]
    image = image[coords[0]:coords[2], coords[3]: coords[1]]
    image = cv2.resize(image,(160, 160))
    encoding = face_recognition.face_encodings(image, known_face_locations=[(0, 160, 160, 0)])[0]
    filter = face_recognition.compare_faces(encodings, encoding, tolerance=0.1)
    if sum(filter) > 0:
        return labels[filter]
    else: return ['Unknown']
    
class ImageUploaderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image to NumPy Array Loader")
        self.setGeometry(100, 100, 500, 700)
        
        layout = QVBoxLayout()
        
        self.image_label = QLabel("Image will be displayed here")
        self.image_label.setStyleSheet("border: 3px solid yellow; background-color: gray;")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(400, 400)
        layout.addWidget(self.image_label)
        
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.setFont(QFont("Arial", 16))
        self.upload_button.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_button)
        
        self.check_button = QPushButton("Check")
        self.check_button.setFont(QFont("Arial", 16))
        self.check_button.setEnabled(False)
        self.check_button.clicked.connect(self.add_random_number)
        layout.addWidget(self.check_button)
        
        self.output_field = QTextEdit()
        self.output_field.setFont(QFont("Arial", 16))
        self.output_field.setReadOnly(True)
        self.output_field.setStyleSheet("border: 3px solid yellow; background-color: #d6a8ff;")
        layout.addWidget(self.output_field)
        
        self.setLayout(layout)
        self.file_path = None
        self.current_image = None
    
    def upload_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.file_path = file_path
            self.current_image = Image.open(file_path)
            self.display_image(self.current_image)
            self.check_button.setEnabled(True)
    
    def display_image(self, image):
        image = image.convert("RGB")
        image = image.resize((400, 400))
        qimage = QImage(image.tobytes(), image.width, image.height, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimage))
    # отвечает за классификацию картинок. никаких рандомных чисел он не присваивает. раньше эта функция была плейсхолдером
    def add_random_number(self):
        if self.file_path:
            try:
                number = str(check_image(self.file_path)[-1])
            except IndexError:
                number = 'Unknown'
            image_with_text = self.current_image.copy()
            draw = ImageDraw.Draw(image_with_text)
            try:
                font = ImageFont.truetype("arial.ttf", 48)
            except IOError:
                font = ImageFont.load_default()
            text = str(number)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            image_width, image_height = image_with_text.size
            text_x = (image_width - text_width) // 2
            text_y = (image_height - text_height) // 2
            draw.text((text_x, text_y), text, font=font, fill="red")
            self.display_image(image_with_text)
            self.current_image = image_with_text
            self.output_field.append(number)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageUploaderApp()
    window.show()
    sys.exit(app.exec())
