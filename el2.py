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


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


class VideoThread(QThread):
    last_pixmap_signal = pyqtSignal(np.ndarray)
    last_pixmap_signal2 = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)
    # emit：detecting/pause/stop/finished/error msg
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)
    send_label_list = pyqtSignal(list)

    send_detect = pyqtSignal()
    send_open = pyqtSignal()
    send_close = pyqtSignal()

    def __del__(self):
        try:
            self.cap.release()
            self.logger.info(f'VideoThread __del__,  cap.release()')
        except:
            pass

    def __init__(self, vi, name):
        super().__init__()

        # self.config = config
        #
        # print(f'{self.config.vi}')
        # print(f'{self.config.font}')

        self.last_img = None
        self._show_flag = False
        self._is_cam_open = False
        self._out_flag = False
        self._set_poi_flag = False
        self._show_poi_flag = False
        self._text_flag = True
        self._detect_flag = False
        self._out_flag = False
        self._change_vi_flag = False  # 영상을 바꾸기 위한 플래그

        self.vi = vi
        self.name = name

        self.load_img()
        # print(f'thread vi : {self.vi}')
        # print(f'thread name : {self.name}')

    def load_img(self):
        default_img = '/home/nvidia/work/auto_el/data/0017.jpg'
        connect_img = '/home/nvidia/work/auto_el/data/connect.jpeg'
        disconnect_img = '/home/nvidia/work/auto_el/data/disconnect.png'
        self.default_img = cv2.imread(default_img, cv2.IMREAD_COLOR)
        self.connect_img = cv2.imread(connect_img, cv2.IMREAD_COLOR)
        self.disconnect_img = cv2.imread(disconnect_img, cv2.IMREAD_COLOR)
        # print(type(self.default_img))
        # self.last_pixmap_signal.emit(self.default_img)

    def run(self):
        print(f'thread start')
        prev_frame_time = 0
        new_frame_time = 0
        label_list = ""
        self.last_pixmap_signal.emit(self.default_img)

        while (True):
            try:
                self.cap = cv2.VideoCapture(self.vi)
                if self.cap.isOpened():
                    print("init cap open success")
                    self._is_cam_open = True
                    self.last_pixmap_signal.emit(self.connect_img)

                    while (True):
                        ret, cv_img = self.cap.read()
                        # new_frame_time = time.time()
                        #
                        # # Calculating the fps
                        # fps = 1 / (new_frame_time - prev_frame_time)
                        # prev_frame_time = new_frame_time
                        # fps = int(fps)
                        # fps = str(fps)
                        if ret:
                            self.last_img = cv_img
                        else:
                            print("cap ret  faule")
                            self._is_cam_open = False
                            self.last_pixmap_signal.emit(self.disconnect_img)
                            break
                else:
                    self._is_cam_open = False
                    print("init cap open fail")
                    self.last_pixmap_signal.emit(self.disconnect_img)

            except Exception as e:
                # self.send_msg.emit('%s' % e)
                print(f"run exception : {e}..... break")
                self._is_cam_open = False
                break

    def cam_open(self):

        print(f'try cam {self.vi} open')
        try:
            cap = cv2.VideoCapture(self.vi)
            print(f'try cam {self.vi} open success')
        except Exception as e:
            print(f"exception : {e}")
            cap = None

        print(f'self.cap type : {type(cap)}')
        print(f'self.cap.isOpened() : {cap.isOpened()}')
        if not cap.isOpened():
            print("Camera open failed!")  # 열리지 않았으면 문자열 출력

    def get_img(self):
        self.last_pixmap_signal.emit(self.last_img)

    def get_img_infer(self):
        return self.last_img


class ImgFrame(QLabel):
    # ResizeSignal = pyqtSignal(int)
    def __init__(self, width, height):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setGeometry(0, 0, width, height)
        self.setFrameShape(QFrame.Box)
        self.setLineWidth(3)
        # self.resize(width, height)
        # self.setText("")
        # front_image = '/home/nvidia/work/auto_el/data/0017.jpg'
        #
        self.pix = QPixmap()
        #
        # self.setPixmap(self.pix)
        self.installEventFilter(self)

    def paintEvent(self, event):
        if not self.pix.isNull():
            size = self.size()
            painter = QPainter(self)

            point = QPoint(0, 0)
            # scaledPix = self.pix.scaled(size, Qt.KeepAspectRatio, transformMode = Qt.FastTransformation)
            scaledPix = self.pix.scaled(size, Qt.IgnoreAspectRatio, transformMode=Qt.FastTransformation)
            # start painting the label from left upper corner
            # point.setX((size.width() - scaledPix.width())/2)
            # point.setY((size.height() - scaledPix.height())/2)
            painter.drawPixmap(point, scaledPix)

    @pyqtSlot(np.ndarray)
    def changePixmap(self, cv_img):
        self.pix = self.convert_cv_qt(cv_img)
        self.repaint()

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(convert_to_Qt_format)


class Cam(QWidget):
    def __init__(self, vi, name):
        super().__init__()
        self.vi = vi
        self.name = name

        # print(f'cam vi : {self.vi}')
        # print(f'cam name : {self.name}')

        self.init_ui()

        self.init_thread()
        # self.init_model()

    def init_ui(self):
        """
        image Widget
        """
        # create the label that holds the image
        self.image_label = ImgFrame(640, 480)

        """
        main Widget
        """
        main_box = QVBoxLayout()
        main_box.addWidget(self.image_label)
        self.setLayout(main_box)

        self.resize(640, 480)

    def contextMenuEvent(self, event):
        contextMenu = QMenu(self)

        # start_act = contextMenu.addAction("Start")
        play_act = contextMenu.addAction("Play")
        infer_act = contextMenu.addAction("Infer")
        action = contextMenu.exec_(self.mapToGlobal(event.pos()))
        if action == play_act:
            self.thread.get_img()
        elif action == infer_act:
            self.thread.get_img_infer()

    def init_thread(self):
        self.thread = VideoThread(self.vi, self.name)
        # self.thread.last_pixmap_signal.connect(self.image_label.changePixmap)
        # self.thread.start()

        # t = VideoThread(value, key)
        # t.last_pixmap_signal.connect(c.image_label.changePixmap)
        # t.default_img()
        # t.start()


class Model(QWidget):
    infer_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, detect_list):
        super().__init__()
        self.fontpath = "/home/nvidia/work/auto_el/data/NanumMyeongjoBold.ttf"
        self.font = ImageFont.truetype(self.fontpath, 40)
        self.yolov5_path = '/home/nvidia/work/yolov5'
        self.v5_pt_path = '/home/nvidia/work/yolov5/yolov5s.pt'
        self.el_pt_path = '/home/nvidia/work/auto_el/data/best.pt'

        # self.detect_list = detect_list

        self.colors = Colors()

        self.load_model(self.yolov5_path, self.el_pt_path)

    @pyqtSlot(np.ndarray)
    def infer(self, cv_img):
        infer_img = None
        if self.model != None:
            results = self.score_frame(cv_img)
            infer_img, label_list = self.plot_boxes(results, cv_img)

        self.infer_pixmap_signal.emit(infer_img)
        # return infer_img

    def infer_img(self, cv_img):
        infer_img = None
        if self.model != None:
            results = self.score_frame(cv_img)
            infer_img, label_list, label_dict = self.plot_boxes(results, cv_img)

            return infer_img, label_list, label_dict

    def put_text(self, frame, text, w, h, color):
        # 한글 표출
        img_pillow = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pillow)
        draw.text((w, h), text, fill=color, font=self.config.font, align="right")

        frame = np.array(img_pillow)  # 다시 OpenCV가 처리가능하게 np 배열로 변환

        return frame

    def load_model(self, yolov5_path, pt_path):

        try:
            # model1 = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            self.model = torch.hub.load(yolov5_path, 'custom', source='local', path=pt_path, force_reload=True)

            self.classes = self.model.names
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f'model type : {type(self.model)}')

        except Exception as e:
            print(f"load_model exception : {e}")

    # @pyqtSlot(np.ndarray)
    # def infer(self, cv_frame):
    #     results = self.score_frame(cv_frame)
    #     infer_img, label_list = self.plot_boxes(results, cv_frame)
    #     self.infer_img_signal.emit(infer_img)
    #     self.infer_label_signal.emit(label_list)

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]



    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        # print(f'plot_boxes  {type(labels)}, {type(cord)}')
        label_list = []
        label_dict ={}
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            name = self.class_to_label(labels[i])
            if name in self.detect_list:
                if row[4] > self.detect_list[name]:
                    x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                        row[3] * y_shape)
                    str = name + ": %0.1f" % row[4]
                    label_dict[name] = "%0.1f" % row[4]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors(int(labels[i])), 2)
                    cv2.putText(frame, str, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.colors(int(labels[i])), 2)
                    label_list.append(str)
                    # if name == 'door_open':
                    #     self.send_open.emit()
                    # elif name == 'door_close':
                    #     self.send_close.emit()
                    # else:
                    #     self.send_detect.emit()
                    # print(f'{name} {row[4]}')
        # print(label_list)
        return frame, label_list, label_dict


class App2(QWidget):
    def __init__(self):
        super().__init__()
        self.el_name = None
        self.down_cam = None
        self.up_cam = None
        self.model = None

        self.config()

        self.init_model()
        self.init_ui()

        self.init_timer()

    def config(self):
        config_file = "/home/nvidia/work/auto_el/el2/config_4.json"
        with open(config_file, 'r') as f:
            self.json_object = json.load(f)

        self.el_name = self.json_object.get("up_detect")


    def init_model(self):
        self.up_model = Model(self.json_object.get("up_detect"))
        self.down_model = Model(self.json_object.get("down_detect"))
        # self.model.detect_list  = self.json_object.get("detect_list")

    def init_timer(self):
        self.tm = QTimer()
        self.tm.setInterval(300)
        self.tm.timeout.connect(self.infer)
        self.tm.start()
        print("init timer")

    def infer(self):
        # print("run...")
        start = time.time()
        # pass
        for i in range(len(self.up_cam)):
            img = self.up_cam[i].thread.get_img_infer()
            img, label_list, label_dict = self.up_model.infer_img(img)
            self.up_text.receive_data(self.up_cam[i].name, label_dict)
            # self.up_text.append(f'{self.up_cam[i].name} : {str(label)}')
            self.up_cam[i].image_label.changePixmap(img)

        for i in range(len(self.down_cam)):
            img = self.down_cam[i].thread.get_img_infer()
            img, label_list, label_dict = self.down_model.infer_img(img)
            self.down_text.receive_data(self.down_cam[i].name, label_dict)
            # self.down_text.append(f'{self.down_cam[i].name} : {str(label)}')
            self.down_cam[i].image_label.changePixmap(img)

        end = time.time()
        print(f'{end - start:.5f} sec')







    def init_ui(self):
        self.setWindowTitle("Auto EL +++++++++")
        # self.resize(1300, 480)

        self.up_cam = []
        self.down_cam = []



        for key, value in self.json_object.get("up").items():
            # print(key, value)
            c = Cam(value, key)
            c.setMinimumWidth(400)

            c.thread.last_pixmap_signal.connect(c.image_label.changePixmap)
            c.thread.start()

            self.up_cam.append(c)
            # self.up_thread.append(t)

        for key, value in self.json_object.get("down").items():
            # print(key, value)
            c = Cam(value, key)
            c.setMinimumWidth(400)

            c.thread.last_pixmap_signal.connect(c.image_label.changePixmap)
            c.thread.start()

            self.down_cam.append(c)
            # self.down_thread.append(t)

        # self.cam1 = Cam()
        # self.cam1.setMinimumWidth(600)
        # self.cam1.setMinimumHeight(400)
        # self.cam2 = Cam()
        # self.cam2.setMinimumWidth(600)
        # self.up_text = QTextEdit()
        self.up_text  = Cont(self.json_object.get("up_delay_time"))
        self.up_text.setMinimumWidth(300)
        self.down_text = Cont(self.json_object.get("down_delay_time"))
        self.down_text.setMinimumWidth(300)
        # self.textedit.resize(433,300)

        up_box = QHBoxLayout()
        down_box = QHBoxLayout()

        for c in self.up_cam:
            up_box.addWidget(c)
        up_box.addWidget(self.up_text)

        for c in self.down_cam:
            down_box.addWidget(c)
        down_box.addWidget(self.down_text)

        layout = QVBoxLayout()
        layout.addLayout(up_box)
        layout.addLayout(down_box)

        self.setLayout(layout)




class Cont(QTextEdit):
    def __init__(self, call_delay_time):
        super().__init__()

        self.door_open = False
        self.called = False
        self.call_delay_time = call_delay_time

        self.last_open_time = time.time()
        self.last_close_time = time.time()
        self.last_detect_time = time.time()
        self.last_call_time = time.time()

    def receive_data(self, name, dict):
        # print(f'{dict}')
        # self.append(f'[{name}] {dict}')
        if "door_open" in dict:
            self.receive_open()
        elif "door_close" in dict:
            self.receive_close()
        elif "wheelchair" in dict:
            self.check_process(name, dict)
        elif "stroller" in dict:
            self.check_process(name, dict)
        elif "silvercar" in dict:
            self.check_process(name, dict)
        elif "scuter" in dict:
            self.check_process(name, dict)

    # def receive_detect(self, ):
    #     # print(f'                          detect')
    #     self.check_process()

    def receive_open(self):

        if self.door_open == False:
            self.append(f'[{self.now_time_str()}]  open')
            self.last_open_time = time.time()
            self.door_open = True
            self.called = False

    def receive_close(self):

        if self.door_open == True:
            self.append(f'[{self.now_time_str()}]  close')
            self.last_close_time = time.time()

            self.door_open = False

    def push_call(self):
        self.append(f'[{self.now_time_str()}]  called el $$ ')
        self.last_call_time = time.time()
        pass

    def now_time_str(self):
        now = time.localtime()
        return time.strftime('%H:%M:%S', now)

    """
    detect시 문이 단혀있으면 호출했었는지 확인하여 처리
    한번 호출후 3초후 재 호출, EL 출발시간 기다린다
    """

    def check_process(self, name, dict):  # 사물이 인식 되었으면
        self.append(f'[{self.now_time_str()}] [{name}] {dict}')
        if self.door_open == True:  # 문이 열려있으면 넘어감
            # print("detect : door_open == True  $$$$$$$   pass")
            pass

        elif self.called == True:  # 문이 닫혀있고, 호출을 하였으면 넘어감
            # print("detect : called == True     $$$$$$$   pass")
            pass

        else:  # 단혀있고, 호출한적이 없으면 호출한다.
            gap = time.time() - self.last_close_time
            if gap > self.call_delay_time:  # 한번 호출후 딜레이 시간 후 재 호출, EL 출발시간 기다린다
                self.append(f' gap :{gap}')
                self.push_call()
                self.called = True


if __name__ == "__main__":
    app = QApplication(sys.argv)

    a = App2()
    a.show()
    sys.exit(app.exec_())
