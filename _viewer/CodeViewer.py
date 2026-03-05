from PyQt5 import QtWidgets, QtGui, QtCore
from qutepart import Qutepart


class CodeViewer(QtWidgets.QMainWindow):
    def __init__(self, file_path, window_title, lang="YAML", parent=None, *super_args, **super_kwargs):
        super().__init__(parent, *super_args, **super_kwargs)
        self.editor = Qutepart(self)
        self.lang = lang
        self.set_new_file(file_path, window_title)
        self.editor.setReadOnly(True)
        self.setCentralWidget(self.editor)
        screen_size = QtWidgets.QApplication.primaryScreen().availableSize()
        self.resize(screen_size.width() // 2, screen_size.height())

    def set_new_file(self, file_path, window_title):
        self.file_path = file_path
        self.editor.text = file_path.read_text()
        self.setWindowTitle(f"Config file preview [{window_title}]")
        self.editor.detectSyntax(language=self.lang)