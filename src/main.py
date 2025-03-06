import sys
import models.model as model
import ui
import os

from config import Config

from PyQt6.QtWidgets import QApplication

# os.environ["QT_QPA_PLATFORM"] = "offscreen"
encodings_path_arc = Config.encodings_path_arc
labels_path_arc = Config.labels_path_arc
encodings_path = Config.encodings_facenet
labels_path = Config.train_labels_facenet

face_net = model.FaceNetModel(encodings_path = encodings_path,
                                 labels_path = labels_path)
arc_face = model.ArcFaceModel(encodings_path = encodings_path_arc,
                                 labels_path = labels_path_arc,
                                 yolo_model_path=Config.yolo_model_path)
app = QApplication(sys.argv)
window = ui.ImageUploaderApp(model = arc_face)
window.show()
print("Приложение запущено")
sys.exit(app.exec())
# --------------------Tkinter version--------------------------
# root = ui.tk.Tk()
# app = ui.ImageUploaderApp(root, model = face_rec_model)
# root.mainloop()