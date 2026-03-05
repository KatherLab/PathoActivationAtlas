from PyQt5 import QtWidgets, QtGui, QtCore
from pathlib import Path

from .GridWidget import GridWidget
from .DataAtlas import DataAtlas
from .DataActgrid import DataActgrid
from .DataClassVis import DataClassVis
from .ThumbnailWidget import ThumbnailWidget
from .CodeViewer import CodeViewer
from .misc import Legend


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.resource_dir = Path("_viewer/_resources")
        self.last_parent_dir = None
        self.code_viewer = None
        self.create_ui()

    def closeEvent(self, event):
        QtWidgets.QApplication.closeAllWindows()

    def create_ui(self):
        self.setObjectName("MainWindow")
        self.setWindowTitle("Viewer")
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

        self.layer_list = QtWidgets.QListWidget(self.sidebar_content)
        self.layer_list.setObjectName("layer_list")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.layer_list.sizePolicy().hasHeightForWidth())
        self.layer_list.setSizePolicy(sizePolicy)
        self.layer_list.setMinimumSize(QtCore.QSize(0, 0))
        self.layer_list.setIconSize(QtCore.QSize(50,50))
        self.layer_list.setViewMode(0) # == list mode
        self.layer_list.setDragDropMode(4) # == InternalMove, aka entries can be moved inside the list widget
        self.layer_list.setAlternatingRowColors(True)
        self.sidebar_area_layout.addWidget(self.layer_list)

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
            QtGui.QIcon(str(self.resource_dir/"file-lines-regular.svg")), "Open new file", self.open_file
        )
        self.open_file_btn.setObjectName("open_file_btn")
        tb_layout = self.options.layout()
        for i in range(tb_layout.count()):
            tb_layout.itemAt(i).setAlignment(QtCore.Qt.AlignLeft)

        self.grid_area = QtWidgets.QWidget(self.central_widget)
        self.grid_area.setObjectName("grid_area")
        self.grid_area_layout = QtWidgets.QGridLayout(self.grid_area)
        self.central_widget.addWidget(self.grid_area)
        self.central_widget.setSizes([int(0.25*self.width()), int(0.75*self.width())])


    def open_file(self):
        if self.last_parent_dir is not None:
            dir = str(self.last_parent_dir)
        else:
            dir = "saved/"
        file_name = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Select atlas/actgrid/class_vis file",
            directory=dir,
            filter="atlas, actgrid or class_vis file (atlas actgrid class_vis)"
        )[0]
        vis_type = Path(file_name).stem
        if vis_type in ["atlas", "actgrid", "class_vis"]:
            try:
                self.last_parent_dir = Path(file_name).parent.parent
            except Exception as e:
                print(f"[MainWindow]: Could not set self.last_parent_dir. See Exception: {e}")
                print("Ignoring and moving on.")
            self.cleanup()
            data_classes = {
                "atlas": DataAtlas,
                "actgrid": DataActgrid,
                "class_vis": DataClassVis
            }
            data = data_classes[vis_type](file_name)
        else:
            return
        if self.code_viewer is None:
            self.code_viewer = CodeViewer(data.marker_file_name, data.exp_name)
        else:
            self.code_viewer.set_new_file(data.marker_file_name, data.exp_name)
        config_btn = self.options.addAction(
            QtGui.QIcon(str(self.resource_dir / "magnifying-glass-solid.svg")),
            "View config file"
        )
        config_btn.triggered.connect(self.code_viewer.show)
        self.thumbnail_widget = ThumbnailWidget(self)
        self.thumbnail_widget.setObjectName("thumbnail_widget")
        self.thumbnail_area_layout.addWidget(self.thumbnail_widget)
        self.grid_widget = GridWidget(self, data)
        self.grid_widget.setObjectName("grid_widget")
        self.grid_area_layout.addWidget(self.grid_widget)
        tb_layout = self.options.layout()
        for i in range(tb_layout.count()):
            tb_layout.itemAt(i).setAlignment(QtCore.Qt.AlignLeft)

    def cleanup(self):
        """To be called before processing a new file, brings everything back to the state before opening any files"""
        first_run = False
        try:
            self.layer_list.currentItemChanged.disconnect()
        except TypeError as e:
            first_run = True
        self.layer_list.clear()
        for child in self.thumbnail_area.children():
            if child.isWidgetType():
                self.thumbnail_area_layout.removeWidget(child)
                child.setParent(None)
                child.deleteLater()
        actions = self.options.actions()
        for act in actions:
            if act.text() != "Open new file":
                self.options.removeAction(act)
                act.deleteLater()
        for child in self.grid_area.children():
            if child.isWidgetType():
                self.grid_area_layout.removeWidget(child)
                child.setParent(None)
                child.deleteLater()
        if not first_run:
            self.grid_widget.overlay_menu.deleteLater()
        self.legend.reset()