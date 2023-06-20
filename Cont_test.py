from PyQt5.QtWidgets import *
# from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtTest import *
import time
import numpy as np
import cv2
import sys
import queue
# import log
import torch
from PIL import Image, ImageFont, ImageDraw
from random import shuffle

import Cont


class Gui(QWidget):

    def __init__(self):
        super().__init__()

        self.cont = Cont.Cont(15)

        self.btn_wheelchair = QPushButton("wheelchair")
        self.btn_wheelchair.clicked.connect(self.detect_wheelchair)

        self.btn_stroller = QPushButton("stroller")
        self.btn_stroller.clicked.connect(self.detect_stroller)

        self.btn_silvercar = QPushButton("silvercar")
        self.btn_silvercar.clicked.connect(self.detect_silvercar)

        self.btn_scuter = QPushButton("scuter")
        self.btn_scuter.clicked.connect(self.detect_scuter)

        self.btn_door_open = QPushButton("door_open")
        self.btn_door_open.clicked.connect(self.open)

        self.btn_door_close = QPushButton("door_close")
        self.btn_door_close.clicked.connect(self.close)

        hbox = QHBoxLayout()

        hbox.addWidget(self.btn_wheelchair)
        hbox.addWidget(self.btn_stroller)
        hbox.addWidget(self.btn_silvercar)
        hbox.addWidget(self.btn_scuter)
        hbox.addWidget(self.btn_door_open)
        hbox.addWidget(self.btn_door_close)

        self.setLayout(hbox)

    def detect_wheelchair(self):
        dict = {"wheelchair": 0.9}
        self.cont.receive_data(dict)

    def detect_stroller(self):
        dict = {"stroller": 0.9}
        self.cont.receive_data(dict)

    def detect_silvercar(self):
        dict = {"silvercar": 0.9}
        self.cont.receive_data(dict)

    def detect_scuter(self):
        dict = {"scuter": 0.9}
        self.cont.receive_data(dict)


    def open(self):
        dict = {"door_open":0.9}
        self.cont.receive_data(dict)

    def close(self):
        dict = {"door_close":0.9}
        self.cont.receive_data(dict)

class App():
    def __init__(self):
        super().__init__()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = Gui()
    gui.show()
    sys.exit(app.exec_())

