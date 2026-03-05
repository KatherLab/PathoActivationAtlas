# launch atlas annotator from here

from PyQt5 import QtWidgets
import sys
#import os
#os.system("pyuic5 -o _viewer/_resources/Ui_MainWindow.py _viewer/_resources/MainWindow.ui")
################################################
from _annotator.MainWindow import MainWindow


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    app.exec_()