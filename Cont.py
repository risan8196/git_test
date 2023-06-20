import time
from PyQt5.QtWidgets import *

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

    def init_ui(self):

    def receive_data(self, dict):
        print(f'{dict}')
        if "door_open" in dict:
            self.receive_open()
        elif "door_close" in dict:
            self.receive_close()
        elif "wheelchair" in dict:
            self.receive_detect()
        elif "stroller" in dict:
            self.receive_detect()
        elif "silvercar" in dict:
            self.receive_detect()
        elif "scuter" in dict:
            self.receive_detect()

    def receive_detect(self):
        # print(f'                          detect')
        self.check_process()

    def receive_open(self):

        if self.door_open == False:
            print(f'[{self.now_time_str()}]  change open')
            self.last_open_time = time.time()
            self.door_open = True
            self.called = False

    def receive_close(self):

        if self.door_open == True:
            print(f'[{self.now_time_str()}]  change close')
            self.last_close_time = time.time()

            self.door_open = False

    def push_call(self):
        print(f'[{self.now_time_str()}]  called el $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ ')
        self.last_call_time = time.time()
        pass

    def now_time_str(self):
        now = time.localtime()
        return time.strftime('%H:%M:%S', now)

    """
    detect시 문이 단혀있으면 호출했었는지 확인하여 처리
    한번 호출후 3초후 재 호출, EL 출발시간 기다린다
    """

    def check_process(self):  # 사물이 인식 되었으면
        if self.door_open == True:  # 문이 열려있으면 넘어감
            # print("detect : door_open == True  $$$$$$$   pass")
            pass

        elif self.called == True:  # 문이 닫혀있고, 호출을 하였으면 넘어감
            # print("detect : called == True     $$$$$$$   pass")
            pass

        else:  # 단혀있고, 호출한적이 없으면 호출한다.
            gap = time.time() - self.last_close_time
            if gap > self.call_delay_time:  # 한번 호출후 딜레이 시간 후 재 호출, EL 출발시간 기다린다
                print(f' gap :{gap}')
                self.push_call()
                self.called = True