import sys
import src.models.model as model
import ui
import os

from PyQt6.QtWidgets import QApplication

# os.environ["QT_QPA_PLATFORM"] = "offscreen"
encodings_path_arc = 'data/encodings(arcface).pkl'
labels_path_arc = 'data/train_labels(arcface).pkl'
encodings_path = 'data/encodings.pkl'
labels_path = 'data/train_labels.pkl'

face_net = model.FaceNetModel(encodings_path = encodings_path,
                                 labels_path = labels_path)
arc_face = model.ArcFaceModel(encodings_path = encodings_path_arc,
                                 labels_path = labels_path_arc,
                                 yolo_model_path="src/models/model.pt")
app = QApplication(sys.argv)
window = ui.ImageUploaderApp(model = arc_face)
window.show()
print("Приложение запущено")
sys.exit(app.exec())
# --------------------Tkinter version--------------------------
# root = ui.tk.Tk()
# app = ui.ImageUploaderApp(root, model = face_rec_model)
# root.mainloop()