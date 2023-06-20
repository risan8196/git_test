from PyQt5.QtWidgets import *
# from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtTest import *
import time
import numpy as np
import cv2
import sys
import json
import queue
# import log
import torch
from PIL import Image, ImageFont, ImageDraw


class Cont_ml(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.btn_open = QPushButton("open")
        self.btn_close = QPushButton("close")
        self.btn_detect = QPushButton("detect")

        box = QHBoxLayout()
        box.addWidget(self.btn_open)
        box.addWidget(self.btn_close)
        box.addWidget(self.btn_detect)
        self.setLayout(box)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    a = Cont_ml()
    a.show()
    sys.exit(app.exec_())