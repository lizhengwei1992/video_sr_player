# -*- coding: utf-8 -*-

from threading import Thread

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import datetime
import cv2
import sys
from sr import sr_process
import pdb
__author__ = "li_zhengwei"
__version__ = "1.00"


class Video_Caputer:
    def __init__(self):
        super(Video_Caputer, self).__init__()
    def init(self, url):
        stream = cv2.VideoCapture(url)
        fps = round(stream.get(cv2.CAP_PROP_FPS))
        size = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),   
                int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frame_all = round(stream.get(cv2.CAP_PROP_FRAME_COUNT))

        ifo = [frame_all, size, fps]

        return stream, ifo



class VideoPlayerGui(QMainWindow):

    def __init__(self, screen, *args, **kwargs):
        super(VideoPlayerGui, self).__init__(*args, **kwargs)      
        # -------------------------------------------------------------------------
        self.VidFrame = QLabel()
        


        self.setCentralWidget(self.VidFrame)
        # Create new action
        playAction = QAction(QIcon('play.png'), '&Play', self)  
        playAction.setShortcut('P')
        playAction.triggered.connect(self.play)

        pauseAction = QAction(QIcon('pause.png'), '&Pause', self)  
        pauseAction.setShortcut('S')
        pauseAction.triggered.connect(self.pause)


        replayVideoAction = QAction(QIcon('RePlay.png'), '&RePlay', self)        
        replayVideoAction.setShortcut('Ctrl+R')
        replayVideoAction.triggered.connect(self.replay)


        srAction = QAction(QIcon('sr.png'), '&SR', self)  
        srAction.setShortcut('SPACE')
        srAction.triggered.connect(self.sr) 

        quitfullscreenAction = QAction(QIcon('full_Screen.png'), '&Full_Screen', self)  
        quitfullscreenAction.setShortcut('ESC')
        quitfullscreenAction.triggered.connect(self.quit_full_screen) 


        openVideoAction = QAction(QIcon('Video.png'), '&Video', self)        
        openVideoAction.setShortcut('Ctrl+O')
        openVideoAction.setStatusTip('Open movie')
        openVideoAction.triggered.connect(self.openFile)

        openModelAction = QAction(QIcon('Model.png'), '&Model', self)        
        openModelAction.setShortcut('Ctrl+M')
        openModelAction.setStatusTip('Open movie')
        openModelAction.triggered.connect(self.openModel)


        # Create exit action
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)        
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.exitCall)

        # Create menu bar and add action


        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')
        fileMenu.addAction(openVideoAction)
        fileMenu.addAction(openModelAction)
        fileMenu.addAction(exitAction)
        controlMenu = menuBar.addMenu('&play') 
        controlMenu.addAction(playAction)
        controlMenu.addAction(pauseAction)
        controlMenu.addAction(replayVideoAction)
        controlMenu.addAction(srAction)
        controlMenu.addAction(quitfullscreenAction)
        # -------------------------------------------------------------------------
        # screen
        self.screen = screen
        
        self.running = False
        self.sr = False
        self.full_screen = False
        self.video_caputer = Video_Caputer()
        
        self.model_url = './model/model_A.pt'
        self.video_url = ''
        self.video_stream = None
        # ifo [frame_all, size[0], size[1], fps]
        self.video_ifo = []   

        self.timers = list()
        self.position = 0
        
        self.sr_time = 0
        self.sr_act = sr_process(self.model_url)


    def setup_timers(self):
        timer = QTimer(self)
        timer.timeout.connect(self.update_frame)
        timer.start(1000.0 / (self.video_ifo[2]))
        self.timers.append(timer)

    def openModel(self):

        self.running = False

        self.model_url, _ = QFileDialog.getOpenFileName(self, "Open model",
                QDir.currentPath())
        self.sr_act = sr_process(self.model_url)

    def openFile(self):
        self.running = False
        self.video_url, _ = QFileDialog.getOpenFileName(self, "Open Video",
                QDir.currentPath())

        # To do : check video
        self.video_stream, self.video_ifo = self.video_caputer.init(self.video_url)

        self.setup_timers()
        self.show_firt_frame()



    def replay(self):
        self.video_stream, self.video_ifo = self.video_caputer.init(self.video_url)
        self.running = True
    def play(self):
        self.running = True
    def pause(self):
        self.running = False


    def quit_full_screen(self):
        self.full_screen = not self.full_screen
        if self.full_screen:
            self.showFullScreen()
        else:
            self.showMaximized()


    def exitCall(self):
        sys.exit(app.exec_())

    def sr(self):
        self.sr = not self.sr


    def show_firt_frame(self):
        ret, frame = self.video_stream.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            w = self.screen.availableGeometry().width()
            h = self.screen.availableGeometry().height()
            
            frame_height, frame_width, _ = frame.shape
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)    
            frame_height, frame_width, _ = frame.shape
            frame = QImage(frame.data,
                                 frame_width,
                                 frame_height,
                                 frame.strides[0],
                                 QImage.Format_RGB888)
 
            self.VidFrame.setPixmap(QPixmap.fromImage(frame))

    def update_frame(self):
        
        if self.running:
            ret, frame = self.video_stream.read()
            if ret:             
                frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_CUBIC)  

                if self.sr:
                    frame, self.sr_time = self.sr_act.compute(frame, (frame.shape[1], frame.shape[0]))
                else:
                    frame_height, frame_width, _ = frame.shape
                    frame = cv2.resize(frame, (frame_width*3, frame_height*3), interpolation=cv2.INTER_CUBIC)      

                if  not self.full_screen:

                    w = self.screen.availableGeometry().width()
                    h = self.screen.availableGeometry().height()
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)   
                else:
                    w = self.screen.size().width()
                    h = self.screen.size().height()
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)
            
                if self.sr:
                    cv2.putText(frame, 'sr_process_time: {:.1f}ms / frame'.format(self.sr_time *1000), (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 2)
                    
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_height, frame_width, _ = frame.shape
                frame = QImage(frame.data,
                                     frame_width,
                                     frame_height,
                                     frame.strides[0],
                                     QImage.Format_RGB888)
      
                self.VidFrame.setPixmap(QPixmap.fromImage(frame))

            else:
                self.running = False


app = QApplication(sys.argv)

screen = app.primaryScreen()
window = VideoPlayerGui(screen)
window.setWindowTitle('ULSee video SR Player')
window.showMaximized()
window.show()
app.exec_()
