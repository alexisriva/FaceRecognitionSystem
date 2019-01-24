import sys
import PyQt5.QtCore as cor
import PyQt5.QtGui as gui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QBoxLayout, QAction, QHBoxLayout, QFileDialog

from proyecto import getImagesAndLabels, recognizePeople

known_face_encodings, known_face_ids = [], []
people_dic = {}

class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.lblFileName1 = QLabel(None, self)
        self.lblFileName2 = QLabel(None, self)
        self.lblFileName3 = QLabel(None, self)
        self.initUI()

    # Function to add widgets to the main window
    def initUI(self):
        boxlayout = QBoxLayout(QBoxLayout.TopToBottom)
        self.setLayout(boxlayout)

        lblWelcome = QLabel('Sistema de Reconocimiento Facial')
        lblWelcome.setFont(gui.QFont("Verdana", 14))

        lblSubtitle = QLabel('Escoger directorio para el entrenamiento:')
        lblSubtitle.setFont(gui.QFont("Verdana", 10))

        btnUploadTrain = QPushButton('Escoger')
        btnUploadTrain.setFixedSize(80, 30)
        btnUploadTrain.setFont(gui.QFont("Verdana", 10))
        btnUploadTrain.clicked.connect(self.openTrainChooser)

        btnTrain = QPushButton('Entrenar')
        btnTrain.setFixedSize(150, 40)
        btnTrain.setFont(gui.QFont("Verdana", 10))
        btnTrain.clicked.connect(self.funcionEntrenar)

        lblSubtitle2 = QLabel('Escoger directorio de imagenes:')
        lblSubtitle2.setFont(gui.QFont("Verdana", 10))

        btnUploadPhotos = QPushButton('Escoger')
        btnUploadPhotos.setFixedSize(80, 30)
        btnUploadPhotos.setFont(gui.QFont("Verdana", 10))
        btnUploadPhotos.clicked.connect(self.openFileChooser)

        lblSubtitle3 = QLabel('Escoger directorio para guardar resultados:')
        lblSubtitle3.setFont(gui.QFont("Verdana", 10))

        btnUploadDir = QPushButton('Escoger')
        btnUploadDir.setFixedSize(80, 30)
        btnUploadDir.setFont(gui.QFont("Verdana", 10))
        btnUploadDir.clicked.connect(self.openResultChooser)

        btnAnalize = QPushButton('Analizar')
        btnAnalize.setFixedSize(150, 40)
        btnAnalize.setFont(gui.QFont("Verdana", 10))
        btnUploadPhotos.clicked.connect(self.funcionAnalisis)

        btnAnalizeVideo = QPushButton('Analizar Live Feed')
        btnAnalizeVideo.setFixedSize(150, 40)
        btnAnalizeVideo.setFont(gui.QFont("Verdana", 10))
        # btnUploadPhotos.clicked.connect(self.funcionAnalisisVideo)

        hbox_TrainChooser = QHBoxLayout()
        hbox_TrainChooser.addWidget(btnUploadTrain)
        hbox_TrainChooser.addWidget(self.lblFileName3)

        hbox_FileChooser = QHBoxLayout()
        hbox_FileChooser.addWidget(btnUploadPhotos)
        hbox_FileChooser.addWidget(self.lblFileName1)

        hbox_ResultsChooser = QHBoxLayout()
        hbox_ResultsChooser.addWidget(btnUploadDir)
        hbox_ResultsChooser.addWidget(self.lblFileName2)

        boxlayout.addWidget(lblWelcome, 0, cor.Qt.AlignCenter)

        boxlayout.addWidget(lblSubtitle, 0, cor.Qt.AlignCenter)
        boxlayout.addLayout(hbox_TrainChooser, 0)
        boxlayout.addWidget(btnTrain, 0, cor.Qt.AlignCenter)
        boxlayout.addWidget(lblSubtitle2, 0, cor.Qt.AlignCenter)
        boxlayout.addLayout(hbox_FileChooser, 0)
        boxlayout.addWidget(lblSubtitle3, 0, cor.Qt.AlignCenter)
        boxlayout.addLayout(hbox_ResultsChooser, 0)
        boxlayout.addWidget(btnAnalize, 0, cor.Qt.AlignCenter)
        boxlayout.addWidget(btnAnalizeVideo, 0, cor.Qt.AlignCenter)
        boxlayout.setAlignment(cor.Qt.AlignCenter)
        boxlayout.setSpacing(20)

        self.resize(800, 500)
        self.setWindowTitle('Reconocimiento Facial')
        self.show()


    def openFileChooser(self):
        fileName = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

        self.lblFileName1.setText(fileName)


    def openResultChooser(self):
        fileName = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

        self.lblFileName2.setText(fileName)


    def openTrainChooser(self):
        fileName = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

        self.lblFileName3.setText(fileName)


    def funcionEntrenar(self):
        data_path = self.lblFileName3.text().strip()
        print(data_path)
        known_face_encodings, known_face_ids, people_dic = getImagesAndLabels(data_path)


    def funcionAnalisis(self):
        test = self.lblFileName1.text().strip()
        output = self.lblFileName2.text().strip()
        recognizePeople(known_face_encodings, known_face_ids, people_dic, test, output)
    

    # def funcionAnalisisVideo(self):
    #     self.lblFileName1.text().strip() # es el link de las imagenes
    #     self.lblFileName1.text().strip() # es el link del directorio para guardar resultados

if __name__ == '__main__':
    app = QApplication(sys.argv)
    root = Main()
    sys.exit(app.exec_())
