import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QTextEdit, QVBoxLayout, QFileDialog
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt
from PIL import Image, ImageDraw, ImageFont
   
class ImageUploaderApp(QWidget):
    def __init__(self, model):
        super().__init__()
        self.initUI()
        self.model = model

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
                number = str(self.model.predict(self.file_path)[-1])
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
    
# --------------------Tkinter version--------------------------
# import tkinter as tk
# from tkinter import filedialog
# from tkinter import ttk
# from PIL import Image, ImageTk, ImageDraw, ImageFont
# import numpy as np
 
# class ImageUploaderApp:
#     def __init__(self, root, model):
#         self.root = root
#         self.model = model
#         self.root.title("Image to NumPy Array Loader")
#         self.root.geometry("500x700")  # Увеличен размер окна
#         self.root.configure(bg="#2b0057")  # Темно-фиолетовый фон
        

#         # Создаем рамку для отображения изображения с желтой рамкой
#         self.image_frame = tk.Label(root, text="Image will be displayed here", bg="gray", fg="white",
#                                     width=50, height=25, bd=5, relief="solid", highlightbackground="yellow", highlightthickness=3)
#         self.image_frame.pack(pady=10, fill="both", expand=True)

#         # Стили для кнопок
#         style = ttk.Style()
#         style.configure("Rounded.TButton",
#                         font=("Arial", 32),  # Увеличенный шрифт кнопок
#                         background="#d6a8ff",  # Светло-фиолетовый цвет
#                         foreground="black",  # Темный текст
#                         padding=15,  # Увеличенные размеры кнопки
#                         borderwidth=2,
#                         relief="flat")
#         style.map("Rounded.TButton",
#                   background=[("active", "#c48cf7")])  # Цвет кнопки при нажатии

#         # Кнопка для загрузки изображения
#         self.upload_button = ttk.Button(root, text="Upload Image", command=self.upload_image, style="Rounded.TButton")
#         self.upload_button.pack(pady=15)

#         # Кнопка для проверки
#         self.check_button = ttk.Button(root, text="Check", command=self.add_random_number, style="Rounded.TButton")
#         self.check_button.pack(pady=15)
#         self.check_button.state(["disabled"])  # Кнопка неактивна, пока изображение не загружено

#         # Текстовое поле для вывода случайных чисел с желтой рамкой
#         self.output_field = tk.Text(root, height=5, width=50, font=("Arial", 32), bg="#d6a8ff", fg="black", bd=5,
#                                     relief="solid", highlightbackground="yellow", highlightthickness=3)
#         self.output_field.pack(pady=10)
#         self.output_field.config(state="disabled")  # Поле изначально заблокировано

#         # Переменные для изображения и массива NumPy
#         self.numpy_image = None
#         self.current_image = None  # PIL.Image для текущего изображения
#         self.file_path = None

#     def upload_image(self):
#         # Открываем диалоговое окно для выбора файла
#         self.file_path = filedialog.askopenfilename(
#             filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;")]
#         )
#         if self.file_path:
#             # Открываем изображение, преобразуем его и сохраняем в виде массива NumPy
#             image = Image.open(self.file_path)
#             self.numpy_image = np.array(image)
#             self.current_image = image  # Сохраняем текущее изображение для дальнейшей обработки

#             # Изменяем размер изображения под рамку
#             image_resized = self._resize_image(image, 512, 512)
#             photo = ImageTk.PhotoImage(image_resized)

#             # Обновляем виджет с изображением
#             self.image_frame.config(image=photo, text="")  # Убираем текст
#             self.image_frame.image = photo  # Сохраняем ссылку на изображение

#             # Активируем кнопку "Проверить"
#             self.check_button.state(["!disabled"])

#     def _resize_image(self, image, max_width, max_height):
#         """Изменяет размер изображения, сохраняя пропорции, чтобы оно вписалось в заданные размеры."""
#         width, height = image.size
#         aspect_ratio = width / height

#         if width > max_width or height > max_height:
#             if aspect_ratio > 1:  # Ширина больше высоты
#                 new_width = max_width
#                 new_height = int(max_width / aspect_ratio)
#             else:  # Высота больше ширины
#                 new_height = max_height
#                 new_width = int(max_height * aspect_ratio)
#         else:
#             new_width, new_height = width, height  # Если изображение меньше, оставляем оригинальные размеры

#         return image.resize((new_width, new_height))

#     def add_random_number(self):
#         if self.file_path:
#             # Генерируем случайное число
#             number = str(self.model.predict(img_path=self.file_path)[-1])

#             # Создаем копию изображения для рисования
#             image_with_text = self.current_image.copy()
#             draw = ImageDraw.Draw(image_with_text)

#             # Определяем шрифт и размеры текста
#             try:
#                 font = ImageFont.truetype("arial.ttf", 48)  # Увеличенный шрифт для текста на изображении
#             except IOError:
#                 font = ImageFont.load_default()  # Если шрифт недоступен

#             # Вычисляем размеры текста с помощью textbbox
#             text = str(number)
#             bbox = draw.textbbox((0, 0), text, font=font)
#             text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

#             # Определяем координаты для центра текста
#             image_width, image_height = image_with_text.size
#             text_x = (image_width - text_width) // 2
#             text_y = (image_height - text_height) // 2
#             draw.text((text_x, text_y), text, font=font, fill="red")

#             # Изменяем размер для отображения и обновляем виджет
#             image_resized = self._resize_image(image_with_text, 512, 512)
#             photo = ImageTk.PhotoImage(image_resized)
#             self.image_frame.config(image=photo)
#             self.image_frame.image = photo

#             # Обновляем текущее изображение
#             self.current_image = image_with_text

#             # Выводим случайное число в текстовое поле
#             self._update_output_field(f"{number}")

#     def _update_output_field(self, text):
#         """Обновляет текстовое поле, добавляя новый текст."""
#         self.output_field.config(state="normal")  # Разрешаем редактирование
#         self.output_field.insert(tk.END, text + "\n")  # Добавляем текст с новой строки
#         self.output_field.see(tk.END)  # Скроллим вниз до последней записи
#         self.output_field.config(state="disabled")  # Блокируем редактирование
