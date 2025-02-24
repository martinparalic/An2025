from pydicom import dcmread
import sys
import os
from strings import * 
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
import numpy as np
import cv2 as cv
from ultralytics import YOLO
import matplotlib.pyplot as plt
from time import time

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.bLock = False
        self.zoom_index = ZOOM_LEVELS.index(1.0)
        self.createGUI()
        self.loadModels()        
        self.loadLastConfig()
        #self.onDetect()
    def createGUI(self):
        self.setWindowTitle(WIN_NAME)
        self.setWindowIcon(QIcon("./icons/ic_main.png"))
        self.createActions()
        self.createMenu()
        self.createToolbar()
        self.createStatusBar()
        self.createLayout()
    def createActions(self):
        self.btnOpen = QAction(OPEN, self)
        self.btnOpen.setIcon(QIcon("./icons/ic_file_open.png"))
        self.btnOpen.triggered.connect(self.onOpen)
        self.btnSave = QAction(SAVE, self)
        self.btnSave.setIcon(QIcon("./icons/ic_file_save.png"))
        self.btnSave.triggered.connect(self.onSave)
        self.btnNext = QAction(NEXT, self)
        self.btnNext.setIcon(QIcon("./icons/ic_next.png"))
        self.btnNext.triggered.connect(self.onNext)
        self.btnPrev = QAction(PREV, self)
        self.btnPrev.setIcon(QIcon("./icons/ic_prev.png"))
        self.btnPrev.triggered.connect(self.onPrev)
        self.btnDetect = QAction(DETECT, self)
        self.btnDetect.setIcon(QIcon("./icons/ic_detect.png"))
        self.btnDetect.triggered.connect(self.onDetect)    
        self.btnCreate  = QAction(CREATE, self)
        self.btnCreate.setIcon(QIcon("./icons/ic_edit.png")) 
        self.btnCreate.triggered.connect(self.onCreateAnnotation)
        self.btnDelete = QAction(DELETE, self)
        self.btnDelete.setIcon(QIcon("./icons/ic_delete.png"))  
        self.btnDelete.triggered.connect(self.onDelete)
        self.btnZoomIn = QAction(ZOOMIN, self)
        self.btnZoomIn.setIcon(QIcon("./icons/ic_zoom_in.png"))
        self.btnZoomIn.triggered.connect(self.onZoomIn)
        self.btnZoomOut = QAction(ZOOMOUT, self)
        self.btnZoomOut.setIcon(QIcon("./icons/ic_zoom_out.png"))
        self.btnZoomOut.triggered.connect(self.onZoomOut)
        self.btnExit = QAction(EXIT, self)
        self.btnExit.setIcon(QIcon("./icons/ic_exit.png"))
        self.btnExit.triggered.connect(self.onExit)
        self.btnHelp = QAction(HELP)
        self.btnHelp.setIcon(QIcon("./icons/ic_help.png"))
        self.btnHelp.triggered.connect(self.onHelp)
        self.btnAbout = QAction(ABOUT)
        self.btnAbout.setIcon(QIcon("./icons/ic_main.png"))
        self.btnAbout.triggered.connect(self.onAbout)
        # ACCELERATORS
        self.btnOpen.setShortcut(QKeySequence("Ctrl+o"))
        self.btnSave.setShortcut(QKeySequence("Ctrl+s"))
        self.btnNext.setShortcut(QKeySequence("Left"))   
        self.btnPrev.setShortcut(QKeySequence("Right"))  
        self.btnDetect.setShortcut(QKeySequence("Tab"))
        self.btnCreate.setShortcut(QKeySequence("Insert"))
        self.btnDelete.setShortcut(QKeySequence("Delete"))
        self.btnZoomIn.setShortcut(QKeySequence("Ctrl++"))
        self.btnZoomOut.setShortcut(QKeySequence("Ctrl+-"))
        self.btnExit.setShortcut(QKeySequence("Alt+x"))
        self.btnHelp.setShortcut(QKeySequence("Ctrl+h"))
        self.btnAbout.setShortcut(QKeySequence("Ctrl+a"))     
    def createMenu(self):
        self.menu = self.menuBar()
        self.menu_file = self.menu.addMenu(FILE)
        self.menu_file.addAction(self.btnOpen)
        self.menu_file.addAction(self.btnSave)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.btnExit)
        self.menu_edit = self.menu.addMenu(EDIT)
        self.menu_edit.addAction(self.btnDetect)
        self.menu_edit.addAction(self.btnCreate)
        self.menu_edit.addAction(self.btnDelete)
        self.menu_view = self.menu.addMenu(VIEW)
        self.menu_view.addAction(self.btnNext)
        self.menu_view.addAction(self.btnPrev)
        self.menu_view.addSeparator()
        self.menu_view.addAction(self.btnZoomIn)
        self.menu_view.addAction(self.btnZoomOut)
        self.menu_help = self.menu.addMenu(HELP)
        self.menu_help.addAction(self.btnHelp)
        self.menu_help.addAction(self.btnAbout)
    def createToolbar(self):
        self.toolbar = QToolBar()
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.toolbar)
        self.toolbar.addAction(self.btnOpen)
        self.toolbar.addAction(self.btnSave)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.btnZoomIn)
        self.toolbar.addAction(self.btnZoomOut)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.btnPrev)
        self.toolbar.addAction(self.btnNext)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.btnDetect)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.btnCreate)
        self.toolbar.addAction(self.btnDelete)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.btnHelp)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.btnExit)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
    def createStatusBar(self):
        status_style = "background-color: yellow; border: 1px solid black;"
        self.report = QLabel()
        self.report.setStyleSheet(status_style)
        self.stZoom = QLabel()
        self.stZoom.setStyleSheet(status_style)
        self.stLevel = QLabel()
        self.stLevel.setStyleSheet(status_style)
        self.status = self.statusBar()
        self.status.addWidget(self.report)
        self.status.addPermanentWidget(self.stZoom)
        self.status.addPermanentWidget(self.stLevel)
    def createLayout(self):
        canvas = QPixmap(VIEW_PORT_WIDTH, VIEW_PORT_HEIGHT)
        canvas.fill(Qt.GlobalColor.darkGray)
        
        self.frame = QLabel()
        self.frame.setPixmap(canvas)
        self.frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame.setMinimumWidth(VIEW_PORT_WIDTH)
        self.frame.setMinimumHeight(VIEW_PORT_HEIGHT)
        self.frame.wheelEvent = self.onMouseWheelEvent
        self.frame.resizeEvent = self.onFrameResizeEvent
        
        self.frame.setMouseTracking(True)        
        self.frame.mouseMoveEvent = self.onMouseMoveEvent 

        self.view_w = self.frame.width()
        self.view_h = self.frame.height()

        self.layout_hor = QHBoxLayout()
        self.layout_ver = QVBoxLayout()
        self.layout_hor.addWidget(self.frame)
        self.layout_hor.addLayout(self.layout_ver)
        
        self.layout_ver.addWidget(QLabel(LB_COLORMAP))
        self.cbColorMap = QComboBox()
        self.cbColorMap.addItems(LB_COLORMAPS)
        self.cbColorMap.currentTextChanged.connect(self.redraw)
        self.layout_ver.addWidget(self.cbColorMap)       

        self.layout_ver.addWidget(QLabel(LB_MODEL))
        self.cbModel = QComboBox()
        self.loadModels()
        self.cbModel.currentIndexChanged.connect(self.setModel)
        self.layout_ver.addWidget(self.cbModel)

        self.cbAnots = QComboBox()
        self.cbAnots.addItems(DEFAULT_ANOTS)
        self.layout_ver.addWidget(self.cbAnots)

        labStatic = QLabel(LB_ANOT)
        self.layout_ver.addWidget(labStatic)

        self.anotlist = QListWidget()
        self.anotlist.setMaximumWidth(200)
        self.anotlist.setMinimumHeight(200)
        self.layout_ver.addWidget(self.anotlist)
        self.widget = QWidget()
        self.widget.setLayout(self.layout_hor)
        self.setCentralWidget(self.widget)
    def loadModels(self):
        model_list = [f for f in os.listdir("./models/")]
        self.cbModel.clear()
        self.cbModel.addItems(model_list)
        self.setModel()
    def changeModel(self, event):
        self.setModel()
    def setModel(self):
        filename = self.cbModel.currentText()
        self.model = YOLO("./models/"+filename, task = "detect")
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Up:
            print("up")
        if event.key() == Qt.Key.Key_Down:
            print("down")
        if event.key() == Qt.Key.Key_Left:
            self.setLevel(self.level-1)
        if event.key() == Qt.Key.Key_Right:
            self.setLevel(self.level+1)
        return super().keyPressEvent(event)
    def onMouseMoveEvent(self, ev):
        x, y = ev.pos().x(), ev.pos().y()
        self.view_rx = x / self.view_w
        self.view_ry = y / self.view_h
    def onFrameResizeEvent(self, event):
        if self.bLock == False:
            self.bLock = True
            QTimer.singleShot(30, self.checkFrameResized)
            print("redraw request")
        else:
            print("chilling")

    def checkFrameResized(self):
        if self.bLock == True:
            QTimer.singleShot(300, self.unlock)
            print("unlock request")
        else:
            print("ignoring")

    def unlock(self):
        self.bLock = False
        print("unchilling", "*"*20)
        QTimer.singleShot(300, self.redraw)

    def onMouseWheelEvent(self, ev):
        val = ev.angleDelta().y()
        sign = int(val / abs(val))
        mod = ev.modifiers()
        # ONLY WHEEL
        if mod == Qt.KeyboardModifier.NoModifier:
            if sign > 0:
                self.onNext()
            else:
                self.onPrev()
        # SHIFT + WHEEL
        if mod == Qt.KeyboardModifier.ShiftModifier:
            self.setLevel(self.level+10*sign)                          
        # CTRL + WHEEL
        if mod == Qt.KeyboardModifier.ControlModifier:
            if sign > 0:
                print(ev.position().x())
                self.onZoomIn()
            else:
                self.onZoomOut()
    def setLevel(self, lev):
        if self.isDicom():
            self.level = lev
            if self.level > len(self.dcm.pixel_array)-1:
                self.level = 0
            if self.level < 0:
                self.level = len(self.dcm.pixel_array)-1
            self.stLevel.setText("Level: {0}".format(self.level))
            self.orig_img = self.dcm.pixel_array[self.level]
            # print("set level")
            self.redraw()
        else:
            self.level = 0
            self.toast("No DCM !!!", 1000)
        self.stLevel.setText("Level: {0}".format(self.level))
    def isDicom(self):
        if hasattr(self, 'dcm'):
            return True
        else:
            return False
    def toast(self, text, duration):
        self.status.showMessage(text, duration)
    def msgbox(self, title, text):
        dlg = QMessageBox(self)
        dlg.setWindowTitle(title)
        dlg.setText(text)
        dlg.exec()
    def clearAll(self):
        # reset status
        # reset graphics & update
        # reset variables
        #del self.dcm
        pass
    def onOpen(self, event, filename=""):
        try:
            if len(filename)==0:
                filename = QFileDialog.getOpenFileName(self, OPEN_FILE, directory= self.last_path, filter = CMD_OPEN_FILE_FILTER, initialFilter = CMD_OPEN_FILE_DEFAULT)
                filename = filename[0]
            # testuje, či si nestlačil cancel
            if len(filename)>0:
                if (filename[-4:]).lower()==".dcm":
                    self.dcm = dcmread(filename)
                    self.setLevel(self.last_level)
                    self.orig_img = self.dcm.pixel_array[self.level]
                    #self.cbColorMap.setEnabled(True)
                else:
                    if self.isDicom():
                        del self.dcm
                    self.cbColorMap.setCurrentText(LB_COLORMAPS[0])
                    #self.cbColorMap.setEnabled(False)
                    tmp = cv.imread(filename)
                    if not self.isGrayscale(tmp):
                        tmp = cv.cvtColor(tmp, cv.COLOR_BGR2RGB)
                    self.orig_img = tmp                    
                    self.setLevel(0)
                self.report.setText(filename)
                self.last_path, self.last_filename = os.path.split(filename)
        except IOError as e:
            self.msgbox(ERROR_TITLE, 
                        "{0}\n[{1} {2}]".format(ERR_FILE_OPEN, e.errno, e.strerror))

        if self.isDicom():
            self.setLevel(int(len(self.dcm.pixel_array)/2))
        QTimer.singleShot(30, self.redraw)
    def onSave(self):
        try:
            filename = QFileDialog.getSaveFileName(self, SAVE_FILE, directory= self.last_path, filter = CMD_SAVE_FILE_FILTER, initialFilter = CMD_SAVE_FILE_DEFAULT)
            print("Saving", filename)
        except:
            self.msgbox(ERROR_TITLE, ERR_FILE_SAVE)
    def onNext(self):
        self.anotlist.clear()
        if self.isDicom():
            self.setLevel(self.level+1)
        else:
            self.toast(ERR_NO_LEVELS, 1000)
    def onPrev(self):
        self.anotlist.clear()
        if self.isDicom():
            self.setLevel(self.level-1)
        else:
            self.toast(ERR_NO_LEVELS, 1000)
    def onDetect(self):
        if self.isDicom() == False:
            self.toast(ERR_NO_INPUT_IMAGE, 1000)
            return
        self.toast(LB_DETECTING, 1000)
        img = self.dcm.pixel_array[self.level]/4.0
        simg = img.astype(np.uint8)
        tmpfile = "./temp/temp.png"
        cv.imwrite(tmpfile, simg)
        result = self.model.predict(tmpfile)
        names = result[0].names
        for box in result[0].boxes:
            rect = box.xyxy
            x1, y1 = int(rect[0][0]), int(rect[0][1])
            x2, y2 = int(rect[0][2]), int(rect[0][3])
            conf = round(float(box.conf),2)
            id = int(box.cls)
            name = "{0} [{1}]".format(names[id], conf)
            self.anotlist.addItem(name)
            print(self.anotlist)
            simg = cv.merge([simg, simg, simg])
            simg = simg.astype(np.uint8)
            self.convert = QImage(simg, self.dcm.Columns, self.dcm.Rows, QImage.Format.Format_RGB888)
            painter = QPainter(self.convert)
            pen = QPen()
            pen.setWidth(1)
            pen.setColor(QColor(0, 200, 0, 255))
            painter.setPen(pen)
            painter.drawRoundedRect(x1, y1, x2-x1, y2-y1, 5, 5)
            font = QFont("Courier", 12)
            painter.setFont(font)
            painter.drawText(x1, y1-10, name)
            painter.end()
            self.frame.setPixmap(QPixmap.fromImage(self.convert))
    def isGrayscale(self, image):
        if len(image.shape) <= 2:
            return True
        else:
            return False
    def prepareImage(self):
        colorMap = self.cbColorMap.currentText()
        if colorMap == "Original":
            if self.isGrayscale(self.orig_img):
                if self.isDicom():
                    divider = 2.0**(self.dcm.BitsStored - 8.0)
                    tmp = self.orig_img / divider 
                    tmp = tmp.astype(np.uint8)
                else:
                    tmp = self.orig_img
                img = cv.cvtColor(tmp, cv.COLOR_GRAY2RGB)
            else:
                img = self.orig_img
            img = img.astype(np.uint8)
        else:    
            cm = plt.get_cmap(colorMap)
            if self.isDicom():
                divider = 2.0**self.dcm.BitsStored 
                tmp = self.orig_img / divider
            else:
                if not self.isGrayscale(self.orig_img):
                    tmp = cv.cvtColor(self.orig_img, cv.COLOR_BGR2GRAY)                     
                tmp = self.orig_img / 256.0
                tmp = tmp[:,:,0]
            tmp = cm(tmp)    
            tmp = (tmp*255.0).astype(np.uint8)
            img = tmp[:,:,:3].astype(np.uint8)

        view_width = self.frame.width()
        view_height = self.frame.height()
        img_height, img_width = img.shape[0:2]
        xs = img_width // 2
        ys = img_height // 2
        
        canvas = np.ones(img.shape, np.uint8)
        roi = img[xs-100:xs+100, ys-100:ys+100,:]
        canvas[xs-100:xs+100, ys-100:ys+100,:] = roi
        return canvas
        #return img.astype(np.uint8)
    def redraw(self):
        if self.bLock:
            return
        t0 = time()
        self.bLock = True
        img = self.prepareImage()
        qimg = QImage(img.data, img.shape[1], img.shape[0], QImage.Format.Format_RGB888)
        # QPixmap.fromImage() sometimes fail during conversion
        pixmap = QPixmap.fromImage(qimg.copy())
        self.frame.setPixmap(pixmap)
        self.bLock = False
        t1 = time()
        print("Elapsed in {0}ms.".format(round(1000*(t1-t0))))
    def onCreateAnnotation(self):
        self.frame.setCursor(QCursor(Qt.CursorShape.CrossCursor))
    def onDelete(self):
        index = self.anotlist.currentIndex().row()
        if (index>=0):
            self.anotlist.takeItem(index)
    def onZoomIn(self):
        self.updateZoom(1)
    def onZoomOut(self):
        self.updateZoom(-1)
        # self.resize(768, 768)
    def updateZoom(self, val):
        if val > 0:
            if self.zoom_index < len(ZOOM_LEVELS)-1:
                self.zoom_index += val
        if val < 0:
            if self.zoom_index > 0:
                self.zoom_index -= 1
        self.redraw()
        self.stZoom.setText("Zoom: {0}%".format(100*ZOOM_LEVELS[self.zoom_index]))
    def onHelp(self):
        #os.system("python -m webbrowser -t {0}".format("file://"+os.getcwd()+"/help/help.html"))
        QTimer.singleShot(30, self.redraw)
    def onAbout(self):
        aboutDlg = AboutDialog()
        aboutDlg.exec()
    def loadLastConfig(self):
        if os.path.exists("./temp") == False:
            print("creating TEMP directory")
            os.mkdir("./temp")
        self.last_path = os.getcwd()
        self.last_filename = ""
        self.last_model = ""
        self.last_level = ""
        colorMap = LB_COLORMAP[0]
        try:
            with open("./temp/last.cfg", "rt") as fid:
                cfg = fid.read()
                cfg = cfg.split('\n')
                for line in cfg:
                    if '=' not in line or len(line)==0:
                        break
                    tag, val = line.split('=')
                    if tag == "PATH":
                        self.last_path = val
                    if tag == "FILE":
                        self.last_filename = val
                    if tag == "MODEL":
                        self.last_model = val
                        pass
                    if tag == "LEVEL":
                        self.last_level = int(val)
                    if tag == "CMAP":
                        colorMap = val
                        pass
                fid.close()
        except:
            pass
        if len(self.last_filename) > 0:
            self.onOpen(any, filename = self.last_path+'/'+self.last_filename)
            self.cbModel.setCurrentText(self.last_model)
            self.cbColorMap.setCurrentText(colorMap)
            self.zoom_index = ZOOM_LEVELS.index(1.0)
            #self.updateZoom(0)
            #self.redraw()
    def onExit(self):
        with open("./temp/last.cfg", "wt") as fid:
            cfg = (
                "PATH={0}\n".format(self.last_path),
                "FILE={0}\n".format(self.last_filename),
                "MODEL={0}\n".format(self.cbModel.currentText()),
                "LEVEL={0}\n".format(self.level),
                "CMAP={0}\n".format(self.cbColorMap.currentText())
            )
            fid.writelines(cfg)
            fid.close()
        quit()

class AboutDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(ABOUT_TITLE)
        img = QPixmap("./icons/ic_main.png")
        ico = img.scaled(48,48)
        lab = QLabel()
        lab.setPixmap(ico)
        lh = QHBoxLayout()
        lh.addWidget(lab)
        lh.addWidget(QLabel(ABOUT_TEXT))
        btn = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btn.accepted.connect(self.close)
        lv = QVBoxLayout()
        lv.addLayout(lh)
        lv.addWidget(btn)
        self.setLayout(lv)
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exec()
if __name__ == "__main__":
    main()