from PyQt5 import QtWidgets, QtGui, QtCore
from pathlib import Path
import sys

from .GridWidget import GridWidget
from .ThumbnailWidget import ThumbnailWidget
from _annotator.misc import Legend


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        try:
            base_dir = Path(sys._MEIPASS)
        except:
            base_dir = Path(".")
        self.resource_dir = base_dir/"_annotator"/"_resources"
        self.last_folder_parent_dir = str(Path("."))
        self.last_desc_parent_dir = str(Path("."))
        self.create_ui()

    def closeEvent(self, event):
        QtWidgets.QApplication.closeAllWindows()

    def create_ui(self):
        self.setObjectName("MainWindow")
        self.setWindowTitle("Annotator")
        screen_size = QtWidgets.QApplication.primaryScreen().availableSize()
        self.resize(screen_size.width() // 2, screen_size.height() // 2)
        self.central_widget = QtWidgets.QSplitter(self)
        self.central_widget.setObjectName("central_widget")
        self.setCentralWidget(self.central_widget)

        self.sidebar_area = QtWidgets.QScrollArea(self.central_widget)
        self.sidebar_area.setObjectName("sidebar_area")
        self.sidebar_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.sidebar_content = QtWidgets.QWidget()
        self.sidebar_area.setWidget(self.sidebar_content)
        self.sidebar_content.setObjectName("sidebar_content")
        self.central_widget.addWidget(self.sidebar_area)
        self.sidebar_area.setWidgetResizable(True)
        self.sidebar_area_layout = QtWidgets.QVBoxLayout(self.sidebar_content)

        self.thumbnail_area = QtWidgets.QWidget(self.sidebar_content)
        self.thumbnail_area.setObjectName("thumbnail_area")
        self.thumbnail_area_layout = QtWidgets.QGridLayout(self.thumbnail_area)
        self.sidebar_area_layout.addWidget(self.thumbnail_area)

        self.legend = Legend()
        self.legend.setObjectName("legend")
        self.sidebar_area_layout.addWidget(self.legend)

        self.option_area = QtWidgets.QWidget(self.sidebar_content)
        self.option_area.setObjectName("option_area")
        self.sidebar_area_layout.addWidget(self.option_area)
        self.option_area_layout = QtWidgets.QVBoxLayout(self.option_area)
        self.options = QtWidgets.QToolBar(self.option_area)
        self.options.setObjectName("options")
        self.options.setOrientation(QtCore.Qt.Vertical)
        self.options.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.option_area_layout.addWidget(self.options)

        self.open_file_btn = self.options.addAction(
            QtGui.QIcon(str(self.resource_dir / "folder-open-regular.svg")),
            "Open image folder\nand class description",
            self.open_folder,
        )
        self.open_file_btn.setObjectName("open_image_folder_btn")

        tb_layout = self.options.layout()
        for i in range(tb_layout.count()):
            tb_layout.itemAt(i).setAlignment(QtCore.Qt.AlignLeft)

        self.grid_area = QtWidgets.QWidget(self.central_widget)
        self.grid_area.setObjectName("grid_area")
        self.grid_area_layout = QtWidgets.QGridLayout(self.grid_area)
        self.central_widget.addWidget(self.grid_area)
        self.central_widget.setSizes([int(0.25 * self.width()), int(0.75 * self.width())])

    def open_folder(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            parent=self, caption="Select atlas image folder", directory=self.last_folder_parent_dir
        )
        if folder_path == "":
            return
        folder_path = Path(folder_path)
        self.last_folder_parent_dir = str(folder_path.parent)

        desc_path = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Select class description file",
            directory=self.last_desc_parent_dir,
            filter="(*.csv)",
        )[0]
        if desc_path == "":
            return
        desc_path = Path(desc_path)
        self.last_desc_parent_dir = str(desc_path.parent)

        self.cleanup()
        self.thumbnail_widget = ThumbnailWidget(self)
        self.thumbnail_widget.setObjectName("thumbnail_widget")
        self.thumbnail_area_layout.addWidget(self.thumbnail_widget)
        self.grid_widget = GridWidget(self, folder_path, desc_path)
        self.grid_widget.setObjectName("grid_widget")
        self.grid_area_layout.addWidget(self.grid_widget)
        tb_layout = self.options.layout()
        for i in range(tb_layout.count()):
            tb_layout.itemAt(i).setAlignment(QtCore.Qt.AlignLeft)
        if not self.grid_widget.options["overlay_active"]:
            self.grid_widget.overlay_btn.trigger()

    def cleanup(self):
        """To be called before processing a new file, brings everything back to the state before opening any files"""
        for child in self.thumbnail_area.children():
            if child.isWidgetType():
                self.thumbnail_area_layout.removeWidget(child)
                child.setParent(None)
                child.deleteLater()
        actions = self.options.actions()
        for act in actions:
            if act.text() != "Open image folder\nand class description":
                self.options.removeAction(act)
                act.deleteLater()
        for child in self.grid_area.children():
            if child.isWidgetType():
                self.grid_area_layout.removeWidget(child)
                child.setParent(None)
                child.deleteLater()
        try:
            self.grid_widget.overlay_menu.deleteLater()
        except:
            pass
        self.legend.reset()
