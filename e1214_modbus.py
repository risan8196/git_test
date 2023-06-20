import time
import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from pyModbusTCP.client import ModbusClient

# TCP auto connect on first modbus request
e1214 = ModbusClient(host="10.128.17.49", port=502, unit_id=1, auto_open=True)

# TCP auto connect on modbus request, close after it
# c = ModbusClient(host="10.128.17.49", auto_open=True, auto_close=True)


"""
DI port  0,  1,  2,  3,  4,   5
reg     [1] [2] [4] [8] [16] [32]
"""


class E1214(QWidget):
    def __init__(self, ip, port):
        super().__init__()
        self.ip = ip
        self.port = port

        self.e1214 = ModbusClient(host=self.ip, port=self.port, unit_id=1, auto_open=True)
        # self.init_ui()
        self.init_zero_relay()

    def init_zero_relay(self):
        self.e1214.write_multiple_coils(0, [0, 0, 0, 0, 0, 0])

    @pyqtSlot(int)
    def push_relay(self, r_no):
        self.e1214.write_single_coil(r_no, 1)
        time.sleep(0.2)
        self.e1214.write_single_coil(r_no, 0)

    #     try:
    #         self.e1214.write_single_coil(r_no, stat)
    #         self.e1214.read_coils(r_no, 1)
    #
    #     except:
    #         return
    @pyqtSlot()
    def get_state(self):
        r = self.e1214.read_coils(0, 6)

    def read_di(self):
        regs = self.e1214.read_input_registers(0x30, 1)
        # # print(type(regs))
        # print(len(regs))
        # print(type(regs[0]))

        state = None
        if regs:
            # print(regs)
            if regs[0] == 0:
                state = f'[DI 0-5] : 000000'
            elif regs[0] == 1:
                state = f'[DI 0-5] : 100000'
            elif regs[0] == 2:
                state = f'[DI 0-5] : 010000'
            elif regs[0] == 4:
                state = f'[DI 0-5] : 001000'
            elif regs[0] == 8:
                state = f'[DI 0-5] : 000100'
            elif regs[0] == 16:
                state = f'[DI 0-5] : 000010'
            elif regs[0] == 32:
                state = f'[DI 0-5] : 000001'
        else:
            state = f'read error'

        return state


class MyWin(QWidget):

    def __init__(self):
        super().__init__()
        self.ip = "10.128.17.49"
        self.port = 502
        self.io = E1214(self.ip, self.port)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("test IO")
        # self.setGeometry(900,100,500,500)

        self.set = "color:blue; background-color: #87CEFA;"
        self.unset = "color:blue; background-color: gray;"

        self.list_di = []
        self.list_r = []
        box_di = QHBoxLayout()
        for i in range(6):
            label = QLabel("DI "+str(i))
            label.setStyleSheet(self.unset)
            label.setMinimumSize(100, 30)
            label.setAlignment(Qt.AlignCenter)
            self.list_di.append(label)
            box_di.addWidget(label)

        box_R = QHBoxLayout()
        for i in range(6):
            label = QPushButton("Rel  "+str(i))
            label.setStyleSheet(self.unset)
            label.setMinimumSize(100, 30)
            label.setObjectName(str(i))
            # label.setAlignment(Qt.AlignCenter)
            label.clicked.connect(self.clicked_r)
            self.list_r.append(label)
            box_R.addWidget(label)

        box_cont = QHBoxLayout()
        btn_read = QPushButton("DI read")
        btn_read.clicked.connect(self.clicked_read)
        # btn_refresh = QPushButton("refresh")
        box_cont.addWidget(btn_read)
        # box_cont.addWidget(btn_refresh)

        box_main = QVBoxLayout()
        box_main.addLayout(box_di)
        box_main.addLayout(box_R)
        box_main.addLayout(box_cont)

        self.setLayout(box_main)

    def clicked_r(self):
        btn = self.sender()
        print(f"{btn.objectName()}")
        self.io.push_relay(int(btn.objectName()))
    def clicked_read(self):
        # print(f'read : {self.io.e1214.read_coils(0, 6)}')
        i =0
        for st in self.io.e1214.read_coils(0, 6):
            print(st)
            if st:
                self.list_di[i].setStyleSheet(self.set)
            else:
                self.list_di[i].setStyleSheet(self.unset)
            i = i+1

if __name__ == "__main__":
    app = QApplication(sys.argv)

    a = MyWin()
    a.show()
    sys.exit(app.exec_())
