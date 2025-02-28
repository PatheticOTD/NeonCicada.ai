import sys
import model
import ui
import os

from PyQt6.QtWidgets import QApplication

# os.environ["QT_QPA_PLATFORM"] = "offscreen"
encodings_path = 'data/encodings.pkl'
labels_path = 'data/train_labels.pkl'

face_rec_model = model.FaceModel(encodings_path = encodings_path,
                                 labels_path = labels_path)
app = QApplication(sys.argv)
window = ui.ImageUploaderApp(model = face_rec_model)
window.show()
print("Приложение запущено")
sys.exit(app.exec())
# --------------------Tkinter version--------------------------
# root = ui.tk.Tk()
# app = ui.ImageUploaderApp(root, model = face_rec_model)
# root.mainloop()