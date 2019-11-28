# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import math
import threading
import pyaudio
from scipy import signal
path = ''
yt = 0
yf = 0
x  = 0
E = 0
Z = 0
def play():
    chunk = 1024  # 2014kb
    wf = wave.open(path, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                    rate=wf.getframerate(), output=True)

    data = wf.readframes(chunk)  # 读取数据

    while True:
        data = wf.readframes(chunk)
        if data == "":
            break
        stream.write(data)
    stream.stop_stream()  # 停止数据流
    stream.close()
    p.terminate()  # 关闭 PyAudio

    print('play函数结束！')


def thread_it(func, *args):
    '''将函数打包进线程'''
    # 创建
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    # 启动
    t.run()

def thread_it_(func, *args):
    '''将函数打包进线程'''
    # 创建
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    # 启动
    t.start()

def catch_number(y, framerate, nframes):
    frh = []
    frl = []
    math_list = [] # 存放频率
    # DTMF编码
    fruquent = [697, 770, 852, 941, 1209, 1336, 1477, 1633]
    math = {(697, 1209): '1', (697, 1336): '2', (697, 1477): '3', (697, 1633): 'A',
            (770, 1209): '4', (770, 1336): '5', (770, 1477): '6', (770, 1633): 'B',
            (852, 1209): '7', (852, 1336): '8', (852, 1477): '9', (852, 1633): 'C',
            (941, 1209): '*', (941, 1336): '0', (941, 1477): '#', (941, 1633): 'D'}
    for fr in fruquent:
        sum1 = sum(y[fr* nframes //framerate-(20* nframes //framerate):fr* nframes //framerate])
        sum2 = sum(y[fr* nframes //framerate:fr * nframes //framerate+(20* nframes //framerate)])
        sumx = sum1 + sum2
        if fr < 1000:
            frl.append([fr, sumx])
        else:
            frh.append([fr, sumx])
    frl = sorted(frl, key=lambda x: x[1])
    frh = sorted(frh, key=lambda x: x[1])
    frl = frl.pop()
    frh = frh.pop()
    if frh[1] < 10:
        return None
    if frl[1] < 10:
        return None
    math_list.append(frl[0])
    math_list.append(frh[0])
    math_list = tuple(math_list)
    for key, value in math.items():
        if key == math_list:
            return value
def sgn(data):
    if data >= 0 :
        return 1
    else:
        return 0

# 基于短时过零率和短时能量的双门限端点检测
def Endpoint_detection(wave_data, energy, Zero) :
    sum = 0
    energyAverage = 0
    for en in energy :
        sum = sum + en
    energyAverage = sum / len(energy)

    sum = 0
    for en in energy[:10] :
        sum = sum + en
    ML = sum / 10
    MH = energyAverage /4             #较高的能量阈值
    ML = (ML + MH) / 2    #较低的能量阈值
    sum = 0
    for zcr in Zero[:100] :
        sum = float(sum) + zcr
    Zs = sum / 100                    #过零率阈值

    A = []
    B = []
    C = []
    print(MH,ML,Zs)

    # 首先利用较大能量阈值 MH 进行初步检测
    flag = 0
    for i in range(len(energy)):
        if len(A) == 0 and flag == 0 and energy[i] > MH :
            A.append(i)
            flag = 1
        elif flag == 0 and energy[i] > MH and i - 21 > A[len(A) - 1]:
            A.append(i)
            flag = 1
        elif flag == 0 and energy[i] > MH and i - 21 <= A[len(A) - 1]:
            A = A[:len(A) - 1]
            flag = 1

        if flag == 1 and energy[i] < MH :
            A.append(i)
            flag = 0
    print("较高能量阈值，计算后的浊音A:" + str(A))

    # 利用较小能量阈值 ML 进行第二步能量检测
    for j in range(len(A)) :
        i = A[j]
        if j % 2 == 1 :
            while i < len(energy) and energy[i] > ML :
                i = i + 1
            B.append(i)
        else :
            while i > 0 and energy[i] > ML :
                i = i - 1
            B.append(i)
    print("较低能量阈值，增加一段语言B:" + str(B))

    # 利用过零率进行最后一步检测
    for j in range(len(B)) :
        i = B[j]
        if j % 2 == 1 :
            while i < len(Zero) and Zero[i] >= 3 * Zs :
                i = i + 1
            C.append(i)
        else :
            while i > 0 and Zero[i] >= 3 * Zs :
                i = i - 1
            C.append(i)
    # 除去帧数小于10的语音段
    for i, data in enumerate(C):
        if i % 2 == 1:
            x = C[i] - C[i-1]
            if x < 10:
                del C[i]
                del C[i-1]

    print("过零率阈值，最终语音分段C:" + str(C))
    count = []
    for data in C:
        count.append(data * 256)
    return count

# path为文件路径
# 输出为x频率信号的自变量，yf2是频率数据，framrate为采样率，nframes为采样点数，yt为单声道的时域数据
def  fftc(path):
    # 打开wav文件 ，open返回一个的是一个Wave_read类的实例，通过调用它的方法读取WAV文件的格式和数据。
    file = wave.open(path, "rb")
    # 一次性返回所有的WAV文件的格式信息，它返回的是一个组元(tuple)：声道数, 量化位数（byte单位）, 采
    # 样频率, 采样点数, 压缩类型, 压缩类型的描述。
    information = file.getparams()
    nchannels, sampwidth, framerate, nframes = information[:4]
    # 读取声音数据，传递一个参数指定需要读取的长度（以取样点为单位）
    str_data = file.readframes(nframes)
    file.close()
    # 将波形数据转换成数组
    # 需要根据声道数和量化单位，将读取的二进制数据转换为一个可以计算的数组
    wave_data = np.fromstring(str_data, dtype=np.short)
    # 将wave_data数组改为2列，行数自动匹配。在修改shape的属性时，需使得数组的总长度不变。
    wave_data.shape = -1, 2
    wave_data = wave_data.T
    yt = wave_data[0]
    yf2 = yt
    x = np.arange(0, len(yf2) / 2) * framerate / nframes
    yf1 = abs(fft(yf2)) / len(yt)  # 归一化处理
    yf2 = yf1[range(int(len(yf2) / 2))]  # 由于对称性，只取一半区间
    return x, yf2, framerate, nframes, yt

#计算短时过零率
def calZero(waveData):
    frameSize = 256
    overLap = 0
    wlen = len(waveData)
    step = frameSize - overLap
    frameNum = math.ceil(wlen/step)
    zcr = np.zeros((frameNum,1))
    for i in range(frameNum):
        curFrame = waveData[np.arange(i*step, min(i*step+frameSize,wlen))]
        curFrame = curFrame - np.mean(curFrame) # zero-justified
        zcr[i] = sum(curFrame[0:-1]*curFrame[1::] <= 0)
    return zcr


#计算短时能量
def calEnergy(wave_data) :
    energy = []
    sum = 0
    for i in range(len(wave_data)) :
        sum = sum + (int(wave_data[i]) ** ui.horizontalSlider_2.value() )
        if (i + 1) % 256 == 0 :
            energy.append(sum)
            sum = 0
        elif i == len(wave_data) - 1 :
            energy.append(sum)
    return energy


#函数实现：通过截取的语音片段，打出数字
#count为截取语音片段的时域信号的采样点数，yt是时域信号
#打印出按键音数字
def continue_math(count,yt,framerate):
    x = []
    math = []
    for i, data in enumerate(count):
        x.append(data)
        if (i + 1) % 2 == 0:
            y1 = yt[x[0]:x[1]]
            yf1 = abs(fft(y1)) / len(y1)  # 归一化处理
            yf2 = yf1[range(int(len(y1) / 2))]  # 由于对称性，只取一半区间
            math_ = catch_number(yf2, framerate, x[1] - x[0])
            x = []
            if math_ == None:
                continue
            math.append(math_)

    return math




def recognition():
    x, yf2, framerate, nframes, yt = fftc(path)
    b, a = signal.butter(11, 0.028, 'highpass')
    yt_updata = signal.filtfilt(b, a, yt)  # data为要过滤的信号
    b, a = signal.butter(11, 0.0687, 'lowpass')
    yt_updata = signal.filtfilt(b, a, yt_updata)  # data为要过滤的信号
    Z = calZero(yt_updata)
    E = calEnergy(yt_updata)
    count = Endpoint_detection(yt_updata,E,Z)
    math = continue_math(count, yt_updata , framerate)
    math_ = ''.join(math)
    QtWidgets.QMessageBox.information(MainWindow,"你输入的数字",math_,QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)


def actionDisplay_frequency_domain_signal():
    plt.xlim((0,2000))
    plt.plot(x, yf)
    plt.show()

def actionDisplay_zero_crossing_signal():
    plt.plot(Z)
    plt.show()

def actionDisplay_E_signal():
    plt.plot(E)
    plt.show()

def actionDisplay_T_signal():
    plt.plot(yt)
    plt.show()


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(500, 600)
        MainWindow.setMinimumSize(QtCore.QSize(500, 600))
        MainWindow.setMaximumSize(QtCore.QSize(500, 600))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(50, 190, 62, 251))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_4 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        self.label_3 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(300, 480, 160, 20))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalSlider.setRange(0,200)
        self.horizontalSlider_2 = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_2.setGeometry(QtCore.QRect(300, 520, 160, 20))
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.horizontalSlider_2.setRange(2, 4)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(250, 210, 113, 30))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(250, 340, 113, 30))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(250, 270, 113, 30))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(250, 400, 113, 30))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(230, 480, 51, 21))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(230, 520, 51, 21))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(300, 460, 54, 12))
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(300, 510, 54, 12))
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(140, 10, 211, 131))
        self.label_9.setStyleSheet("image: url(:/新前缀/apple.gif);")
        self.label_9.setText("")
        self.label_9.setObjectName("label_9")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(150, 100, 90, 50))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(270, 100, 90, 50))
        self.pushButton_2.setObjectName("pushButton_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 500, 23))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        MainWindow.setMenuBar(self.menubar)
        self.actionOpen_file = QtWidgets.QAction(MainWindow)
        self.actionOpen_file.setObjectName("actionOpen_file")
        self.actionExit = QtWidgets.QAction(MainWindow)


        self.actionExit.setObjectName("actionExit")

        self.actionShow_chart = QtWidgets.QAction(MainWindow)
        self.actionShow_chart.setObjectName("actionShow_chart")
        self.actionAsd_a = QtWidgets.QAction(MainWindow)
        self.actionAsd_a.setObjectName("actionAsd_a")
        self.actionDisplay_zero_crossing_signal = QtWidgets.QAction(MainWindow)
        self.actionDisplay_zero_crossing_signal.setObjectName("actionDisplay_zero_crossing_signal")
        self.actionDisplay_frequency_domain_signal = QtWidgets.QAction(MainWindow)
        self.actionDisplay_frequency_domain_signal.setObjectName("actionDisplay_frequency_domain_signal")
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionOpen_file)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menuView.addAction(self.actionShow_chart)
        self.menuView.addAction(self.actionDisplay_frequency_domain_signal)
        self.menuView.addAction(self.actionAsd_a)
        self.menuView.addAction(self.actionDisplay_zero_crossing_signal)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())

        self.retranslateUi(MainWindow)
        self.link(MainWindow)
        self.label_7.setNum(0)
        self.label_8.setNum(2)
        self.horizontalSlider_2.valueChanged['int'].connect(self.label_8.setNum)
        self.horizontalSlider.valueChanged['int'].connect(self.label_7.setNum)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def link(self,MainWindow):
        #actionOpen_file与槽函数相关联
        self.actionOpen_file.triggered.connect(lambda: self.FileDialog(MainWindow))
        #actionExit与槽函数相关联
        self.actionExit.triggered.connect(MainWindow.close)
        # pushButton(播放)与槽函数相关联
        self.pushButton.clicked.connect(lambda :thread_it_(play))
        # pushButton(识别)与槽函数相关联
        self.pushButton_2.clicked.connect(lambda :thread_it(recognition))
        self.actionDisplay_frequency_domain_signal.triggered.connect(actionDisplay_frequency_domain_signal)
        self.actionDisplay_zero_crossing_signal.triggered.connect(actionDisplay_zero_crossing_signal)
        self.actionAsd_a.triggered.connect(actionDisplay_E_signal)
        self.actionShow_chart.triggered.connect(actionDisplay_T_signal)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_4.setText(_translate("MainWindow", "时长："))
        self.label_3.setText(_translate("MainWindow", "频率："))
        self.label.setText(_translate("MainWindow", "通道数："))
        self.label_2.setText(_translate("MainWindow", "点数："))
        self.label_5.setText(_translate("MainWindow", "高阀值："))
        self.label_6.setText(_translate("MainWindow", "倍数："))
        self.pushButton.setText(_translate("MainWindow", "播放"))
        self.pushButton_2.setText(_translate("MainWindow", "识别"))
        self.menuFile.setTitle(_translate("MainWindow", "file"))
        self.menuView.setTitle(_translate("MainWindow", "signal"))
        self.actionOpen_file.setText(_translate("MainWindow", "open  file"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionShow_chart.setText(_translate("MainWindow", "Display time domain signal"))
        self.actionAsd_a.setText(_translate("MainWindow", "Display energy signal"))
        self.actionDisplay_zero_crossing_signal.setText(_translate("MainWindow", "Display zero crossing signal"))
        self.actionDisplay_frequency_domain_signal.setText(_translate("MainWindow", "Display frequency domain signal"))


    def FileDialog(self,MainWindow):
        fileName1, filetype = QtWidgets.QFileDialog.getOpenFileName(MainWindow,"选取文件","./",
                                                          "音频文件 (*.wav);;Text Files (*.txt)")
        if fileName1 == '':
            return None
        global path
        global yt, yf, x, E ,Z
        path = fileName1
        file = wave.open(path, "rb")
        # 一次性返回所有的WAV文件的格式信息，它返回的是一个组元(tuple)：声道数, 量化位数（byte单位）, 采
        # 样频率, 采样点数, 压缩类型, 压缩类型的描述。
        params = file.getparams()
        self.nchannels, self.sampwidth, self.framerate, self.nframes = params[:4]
        x, yf, framerate, nframes, yt = fftc(path)
        E = calEnergy(yt)
        Z = calZero(yt)
        self.lineEdit.setText(str(round((1/self.framerate) * self.nframes,2)) + "秒")
        self.lineEdit_3.setText(str(self.framerate) + "HZ")
        self.lineEdit_2.setText(str(self.nchannels))
        self.lineEdit_4.setText(str(self.nframes))
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
