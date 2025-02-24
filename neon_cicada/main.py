import sys
import model
import ui
from PyQt6.QtWidgets import QApplication

encodings_path = 'data/encodings.pkl'
labels_path = 'data/train_labels.pkl'

face_rec_model = model.FaceModel(encodings_path = encodings_path,
                                 labels_path = labels_path)

app = QApplication(sys.argv)
window = ui.ImageUploaderApp(model = face_rec_model)
window.show()
sys.exit(app.exec())