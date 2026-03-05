from PyQt5 import QtWidgets, QtGui, QtCore

from _creator import utils
from .misc import InteractiveQGraphicsView
from .GridLayerCellWidget import GridLayerCellWidget

class GridLayerWidget(QtWidgets.QWidget):
    relative_scene_rect_signal = QtCore.pyqtSignal("QRectF")
    def __init__(self, layer_name, grid_widget, parent=None, *super_args, **super_kwargs):
        super().__init__(parent, *super_args, **super_kwargs)
        self.data = grid_widget.data
        self.grid_widget = grid_widget
        self.layer_name = layer_name
        self.cell_size = self.data.tile_sizes[self.layer_name]["orig"]
        self.displayed_cell_size = self.cell_size

        # setup scene_grid
        self.setMinimumSize(QtCore.QSize(0, 0))
        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        self.setLayout(layout)
        self.scene = QtWidgets.QGraphicsScene()
        self.view = InteractiveQGraphicsView()
        self.view.setScene(self.scene)
        self.layout().addWidget(self.view)
        form = QtWidgets.QWidget()
        form.setStyleSheet("background-color:white")
        form.resize(0,0)
        self.scene.addWidget(form)
        self.scene_grid = QtWidgets.QGridLayout()
        form.setLayout(self.scene_grid)
        self.scene_grid.setContentsMargins(0, 0, 0, 0)
        self.scene_grid.setSpacing(0)
        self.scene_grid.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)  # ensures adaptable cell size
        self.view.zoom_signal.connect(self.update_displayed_cell_size)
        self.view.viewport_to_scene_signal.connect(self.emit_relative_scene_rect)
        self.relative_scene_rect_signal.connect(self.grid_widget.main_window.thumbnail_widget.receive_new_viewport)

        # fill scene_grid
        for (y, x), scale in self.data.tile_scales[layer_name].items():
            grid_item = GridLayerCellWidget(layer_name, (y,x), self.grid_widget, self)
            self.scene_grid.addWidget(grid_item, y, x)
        for c in range(self.data.num_cells[layer_name]):
            self.scene_grid.setColumnMinimumWidth(c, self.cell_size.width())
            self.scene_grid.setRowMinimumHeight(c, self.cell_size.height())

    def get_scene_img(self):
        sceneSize = self.scene.sceneRect().size().toSize()
        maxDim = 32767
        if sceneSize.width() > maxDim or sceneSize.height() > maxDim:
            sceneSize.scale(maxDim, maxDim, QtCore.Qt.KeepAspectRatio)
        img = QtGui.QImage(sceneSize, QtGui.QImage.Format_ARGB32)
        img.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(img)
        self.scene.render(painter)
        painter.end()
        return img

    ######### slots ###########
    def update_displayed_cell_size(self, zoom_factor):
        self.displayed_cell_size = self.cell_size * zoom_factor

    def emit_relative_scene_rect(self, current_scene_rect):
        # crop current_scene_rect to fit inside the scene's sceneRect
        scene_rect = self.scene.sceneRect()
        current_scene_rect.setTop(max(current_scene_rect.top(), scene_rect.top()))
        current_scene_rect.setBottom(min(current_scene_rect.bottom(), scene_rect.bottom()))
        current_scene_rect.setLeft(max(current_scene_rect.left(), scene_rect.left()))
        current_scene_rect.setRight(min(current_scene_rect.right(), scene_rect.right()))
        # scale current_scene_rects coordinates to [0,1]
        retval = QtCore.QRectF(
            QtCore.QPointF(
                utils.map_to_range(current_scene_rect.left(), scene_rect.left(), scene_rect.right(), 0.0, 1.0),
                utils.map_to_range(current_scene_rect.top(), scene_rect.top(), scene_rect.bottom(), 0.0, 1.0)
            ),
            QtCore.QPointF(
                utils.map_to_range(current_scene_rect.right(), scene_rect.left(), scene_rect.right(), 0.0, 1.0),
                utils.map_to_range(current_scene_rect.bottom(), scene_rect.top(), scene_rect.bottom(), 0.0, 1.0)
            )
        )
        self.relative_scene_rect_signal.emit(retval)