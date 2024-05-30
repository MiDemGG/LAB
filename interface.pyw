import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import numpy as np
from PIL import Image
from keras.models import load_model

class PhotoClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Фото классификатор")
        self.model = load_model(r"C:\Users\midem\Desktop\project_4_sem\my_inception_model.h5")
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.load_button = QPushButton("Выбрать фото")
        self.load_button.clicked.connect(self.load_image)
        self.classify_button = QPushButton("Классифицировать")
        self.classify_button.clicked.connect(self.classify_image)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.load_button)
        layout.addWidget(self.classify_button)
        self.setLayout(layout)

    def load_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Выбрать фото", "", "Файлы изображений (*.png *.jpg *.jpeg)")
        if filename:
            self.image_path = filename
            self.update_image()

    def update_image(self):
        pixmap = QPixmap(self.image_path)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)
        self.image_label.show()

    def classify_image(self):
        if hasattr(self, 'image_path'):
            image = Image.open(self.image_path)
            image = image.resize((224, 224))
            np_image = np.array(image) / 255.0
            np_image = np.expand_dims(np_image, axis=0)
            prediction = self.model.predict(np_image)
            class_names = ['Димандре', 'Игарь', 'Создатель', 'Сава', 'Максончик', 'Ваня']
            print(prediction)
            result = class_names[np.argmax(prediction)]
            QMessageBox.information(self, "Результат", f"Модель предсказывает: {result}")
        else:
            QMessageBox.critical(self, "Ошибка", "Сначала загрузите фотографию")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    classifier_app = PhotoClassifierApp()
    classifier_app.show()
    sys.exit(app.exec_())
