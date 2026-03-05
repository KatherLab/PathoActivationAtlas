from __future__ import annotations
from PyQt5 import QtWidgets, QtGui, QtCore
import cachetools
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
from pathlib import Path

from _annotator.misc import *
from _annotator.GridCellWidget import GridCellWidget

if TYPE_CHECKING:
    from _annotator.MainWindow import MainWindow
    from pathlib import Path


class GridWidget(QtWidgets.QWidget):
    relative_scene_rect_signal = QtCore.pyqtSignal("QRectF")
    annotation_data_to_cells_signal = QtCore.pyqtSignal()
    annotation_data_from_cells_signal = QtCore.pyqtSignal()

    def __init__(
        self,
        main_window: MainWindow,
        folder_path: Path,
        desc_path: Path,
        tile_cache_size: str = "8 GB",
        parent=None,
        *super_args,
        **super_kwargs,
    ):
        super().__init__(parent, *super_args, **super_kwargs)
        self.main_window = main_window
        self.folder_path = folder_path
        self.desc_path = desc_path
        self.main_window.setWindowTitle(f"{self.folder_path.name} | {self.desc_path.name}")
        self.options = {"overlay_opacity": 1.0, "overlay_active": False, "cls_idx": -1}

        # figure out various info about the atlas grid
        self.div_lvls = []
        for p in self.folder_path.glob("div_*"):
            split = p.name.split("_")
            self.div_lvls.append(int(split[1]))
        self.num_x, self.num_y = 0, 0
        for p in folder_path.iterdir():
            if p.suffix == ".png":
                y, x = list(map(int, p.stem.split("_")))
                if y + 1 > self.num_y:
                    self.num_y = y + 1
                if x + 1 > self.num_x:
                    self.num_x = x + 1
        #self.num_cells = max(self.num_x, self.num_y)
        self.tile_sizes = OrderedDict()
        self.tile_sizes["orig"] = QtGui.QPixmap(
            str(next(p for p in self.folder_path.iterdir() if p.suffix == ".png"))
        ).size()
        for lvl in self.div_lvls:
            self.tile_sizes[f"div_{lvl}"] = QtGui.QPixmap(
                str(next(p for p in (self.folder_path / f"div_{lvl}").iterdir() if p.suffix == ".png"))
            ).size()
        self.class_names = pd.read_csv(self.desc_path)
        self.class_names = self.class_names[self.class_names["Name"] != "???"]["Name"].values

        # set up pixmap_cache
        cache_factors = self.div_lvls[:]
        for i in range(len(cache_factors)):
            # space requirement decreases quadratically with scale factor
            cache_factors[i] = cache_factors[i] ** 2
        cache_sizes = divide_cache_size_free_divisors(tile_cache_size, cache_factors)
        self.pixmap_cache = OrderedDict(
            [("orig", cachetools.LRUCache(maxsize=cache_sizes[0], getsizeof=get_size_of_pixmap))]
            + [
                (f"div_{lvl}", cachetools.LRUCache(maxsize=cache_sizes[i + 1], getsizeof=get_size_of_pixmap))
                for i, lvl in enumerate(self.div_lvls)
            ]
        )

        # pre-create pixmaps to use for overlays
        overlay_colors = [
            np.array([c[0] * 255, c[1] * 255, c[2] * 255], dtype=np.uint8)
            for i, c in enumerate(
                plt.cm.Set1.colors + plt.cm.Set2.colors + plt.cm.Set3.colors + plt.cm.Dark2.colors
            )
            if i < len(self.class_names)
        ]
        num_colors = len(overlay_colors)
        overlay_pixels = np.zeros((num_colors, 1, 1, 3), dtype=np.uint8)
        for i, c in enumerate(overlay_colors):
            overlay_pixels[i, 0, 0] = c
        self.overlay_pixmaps = [
            QtGui.QPixmap.fromImage(QtGui.QImage(overlay_pixels[i], 1, 1, QtGui.QImage.Format_RGB888))
            for i in range(num_colors)
        ]
        unknown_pixmap = QtGui.QPixmap(QtCore.QSize(1, 1))
        unknown_pixmap.fill(QtCore.Qt.transparent)
        self.main_window.legend.select_cls_idx_signal.connect(self.update_selected_cls_idx)
        self.main_window.legend.reset(
            class_info_to_pixmap={
                **{(i, cls_name): self.overlay_pixmaps[i] for i, cls_name in enumerate(self.class_names)},
                (-1, "???"): unknown_pixmap,
            }
        )

        # inner grid stuff - formerly in GridLayerWidget
        self.displayed_cell_size = self.tile_sizes["orig"]
        self.setMinimumSize(QtCore.QSize(0, 0))
        layout = QtWidgets.QGridLayout()
        layout.setObjectName("grid_widget_layout")
        layout.setContentsMargins(4, 4, 4, 4)
        # QtCore.QObjectCleanupHandler().add(self.layout())
        self.setLayout(layout)
        self.scene = QtWidgets.QGraphicsScene()
        self.view = InteractiveQGraphicsView()
        self.view.setScene(self.scene)
        self.layout().addWidget(self.view)
        form = QtWidgets.QWidget()
        form.setStyleSheet("background-color:white")
        form.resize(0, 0)
        self.scene.addWidget(form)
        self.scene_grid = QtWidgets.QGridLayout()
        form.setLayout(self.scene_grid)
        self.scene_grid.setContentsMargins(0, 0, 0, 0)
        self.scene_grid.setSpacing(0)
        self.scene_grid.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.view.zoom_signal.connect(self.update_displayed_cell_size)
        self.view.paint_event_signal.connect(self.maybe_update_thumbnail)
        self.view.right_btn_down_move_signal.connect(self.change_cls_from_scene_coord)
        self.view.viewport_to_scene_signal.connect(self.emit_relative_scene_rect)
        self.relative_scene_rect_signal.connect(self.main_window.thumbnail_widget.receive_new_viewport)
        self.annotation_data = dict()
        for p in folder_path.iterdir():
            if p.suffix == ".png":
                y, x = list(map(int, p.stem.split("_")))
                grid_item = GridCellWidget(grid_widget=self, pos=(y, x))
                self.annotation_data_to_cells_signal.connect(grid_item.receive_annotated_cls)
                self.annotation_data_from_cells_signal.connect(grid_item.send_annotated_cls)
                self.scene_grid.addWidget(grid_item, y, x)
                grid_item.update()
        for x in range(self.num_x):
            self.scene_grid.setColumnMinimumWidth(x, self.tile_sizes["orig"].width())
        for y in range(self.num_y):
            self.scene_grid.setRowMinimumHeight(y, self.tile_sizes["orig"].height())

        self.main_window.thumbnail_widget.mouse_drag_signal.connect(self.update_relative_view_center)
        self.main_window.thumbnail_widget.mouse_scroll_signal.connect(self.update_view_scroll)

        # set up the entries in options
        # Save annotations
        save_annotations_btn = self.main_window.options.addAction(
            QtGui.QIcon(str(self.main_window.resource_dir / "floppy-disk-regular.svg")),
            "Save current annotations [Ctrl+S]",
            self.save_annotation_data
        )
        save_annotations_btn.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        # Load annotations
        load_annotations_btn = self.main_window.options.addAction(
            QtGui.QIcon(str(self.main_window.resource_dir/"folder-open-regular.svg")),
            "Open previous annotations [Ctrl+O]",
            self.load_annotation_data
        )
        load_annotations_btn.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        # Overlay opacity
        opacity_slider = SliderWithTitle("Overlay opacity", QtCore.Qt.Horizontal, 0, 1)
        opacity_slider.value_changed_signal.connect(self.update_overlay_opacity)
        self.main_window.options.addWidget(opacity_slider)
        # Overlay toggle
        self.overlay_btn = self.main_window.options.addAction(
            QtGui.QIcon(str(self.main_window.resource_dir / "layer-group-solid.svg")),
            "Toggle annotation overlay [T]",
            self.toggle_annotation_overlay,
        )
        self.overlay_btn.setShortcut(QtGui.QKeySequence("T"))
        self.overlay_btn.setCheckable(True)
        self.options["overlay_active"] = self.overlay_btn.isChecked()
        # Viewport to image file
        viewport_to_image_btn = self.main_window.options.addAction(
            QtGui.QIcon(str(self.main_window.resource_dir / "crop-simple-solid.svg")),
            "Export viewport",
            self.viewport_to_image_file,
        )
        # Scene to image file
        scene_to_image_btn = self.main_window.options.addAction(
            QtGui.QIcon(str(self.main_window.resource_dir / "panorama-solid.svg")),
            "Export scene",
            self.scene_to_image_file,
        )

    def get_scene_img(self, maxDim=32767):
        sceneSize = self.scene.sceneRect().size().toSize()
        if sceneSize.width() > maxDim or sceneSize.height() > maxDim:
            sceneSize.scale(maxDim, maxDim, QtCore.Qt.KeepAspectRatio)
        img = QtGui.QImage(sceneSize, QtGui.QImage.Format_ARGB32)
        if img.isNull():
            return img
        img.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(img)
        self.scene.render(painter)
        painter.end()
        return img

    ########## slots ##########
    def load_annotation_data(self):
        load_path = QtWidgets.QFileDialog.getOpenFileName(
            parent=self.main_window,
            caption="Load annotations from...",
            directory=str(self.folder_path.parent),
            filter="(*.csv)"
        )[0]
        if load_path == "":
            return
        load_path = Path(load_path)
        load_data = pd.read_csv(load_path)
        load_data = load_data[load_data["folder_name"] == self.folder_path.name]
        ys = load_data["y_coord"].values
        xs = load_data["x_coord"].values
        cls_idxs = load_data["cls_idx"].values
        self.annotation_data = dict()
        for y, x, cls_idx in zip(ys, xs, cls_idxs):
            self.annotation_data[(int(y), int(x))] = int(cls_idx)
        self.annotation_data_to_cells_signal.emit()
        self.refresh_views(False)

    def save_annotation_data(self):
        self.annotation_data_from_cells_signal.emit()
        save_path = QtWidgets.QFileDialog.getSaveFileName(
            parent=self.main_window,
            caption="Save annotations as or append annotations to...",
            directory=str(self.folder_path.parent / f"{self.folder_path.name}.csv"),
        )[0]
        if save_path == "":
            return
        save_path = Path(save_path)
        if save_path.suffix != ".csv":
            save_path = save_path.with_suffix(".csv")
        save_data = {"folder_name": [], "y_coord": [], "x_coord": [], "cls_idx": [], "cls_name": []}
        for (y, x), cls_idx in self.annotation_data.items():
            save_data["folder_name"].append(self.folder_path.name)
            save_data["y_coord"].append(y)
            save_data["x_coord"].append(x)
            save_data["cls_idx"].append(cls_idx)
            if cls_idx == -1:
                save_data["cls_name"].append("???")
            else:
                save_data["cls_name"].append(self.class_names[cls_idx])
        save_data = pd.DataFrame(save_data)
        save_data.to_csv(save_path, index=False)

    def update_selected_cls_idx(self, new_val):
        self.options["cls_idx"] = new_val

    def update_overlay_opacity(self, new_val):
        self.options["overlay_opacity"] = new_val
        self.refresh_views()

    def update_displayed_cell_size(self, zoom_factor):
        self.displayed_cell_size = self.tile_sizes["orig"] * zoom_factor

    def change_cls_from_scene_coord(self, scene_coords: QtCore.QPointF):
        # scene_coords to grid y and x positions
        scene_rect = self.scene.sceneRect()
        grid_x = min(int(map_to_range(
            val=scene_coords.x(),
            old_min=scene_rect.left(),
            old_max=scene_rect.right(),
            new_min=0,
            new_max=self.num_x)), self.num_x-1)
        grid_y = min(int(map_to_range(
            val=scene_coords.y(),
            old_min=scene_rect.top(),
            old_max=scene_rect.bottom(),
            new_min=0,
            new_max=self.num_y)), self.num_y-1)
        item = self.scene_grid.itemAtPosition(grid_y, grid_x)
        if item is not None:
            grid_cell_widget = item.widget()
            grid_cell_widget.update_annotated_cls()

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
                map_to_range(
                    current_scene_rect.left(), scene_rect.left(), scene_rect.right(), 0.0, 1.0
                ),
                map_to_range(current_scene_rect.top(), scene_rect.top(), scene_rect.bottom(), 0.0, 1.0),
            ),
            QtCore.QPointF(
                map_to_range(
                    current_scene_rect.right(), scene_rect.left(), scene_rect.right(), 0.0, 1.0
                ),
                map_to_range(
                    current_scene_rect.bottom(), scene_rect.top(), scene_rect.bottom(), 0.0, 1.0
                ),
            ),
        )
        self.relative_scene_rect_signal.emit(retval)

    def toggle_annotation_overlay(self):
        self.options["overlay_active"] = not self.options["overlay_active"]
        self.refresh_views()

    def maybe_update_thumbnail(self):
        if self.main_window.thumbnail_widget.pixmap_base is None:
            scene_img = self.get_scene_img(500)
            self.main_window.thumbnail_widget.update_base_pixmap(QtGui.QPixmap.fromImage(scene_img))

    def refresh_views(self, repaint=True):
        if repaint:
            self.scene.update()
        scene_img = self.get_scene_img(500)
        if scene_img.isNull():
            return
        self.main_window.thumbnail_widget.update_base_pixmap(QtGui.QPixmap.fromImage(scene_img))

    def update_relative_view_center(self, rel_view_center):
        scene_rect = self.scene.sceneRect()
        scene_view_center = QtCore.QPointF(
            map_to_range(rel_view_center.x(), 0.0, 1.0, scene_rect.left(), scene_rect.right()),
            map_to_range(rel_view_center.y(), 0.0, 1.0, scene_rect.top(), scene_rect.bottom()),
        )
        self.view.centerOn(scene_view_center)

    def update_view_scroll(self, angle_delta):
        self.view._wheelEvent(angle_delta)

    def scene_to_image_file(self):
        file_name = QtWidgets.QFileDialog.getSaveFileName(
            parent=self.main_window,
            caption="Save scene image as...",
            directory=str(self.folder_path.parent / "scene.png"),
        )[0]
        if file_name == "":
            return
        p = Path(file_name)
        if p.suffix == "":
            file_name = str(p.with_suffix(".png"))
        qimg = self.get_scene_img()
        qimg.save(file_name)

    def viewport_to_image_file(self):
        file_name = QtWidgets.QFileDialog.getSaveFileName(
            parent=self.main_window,
            caption="Save viewport image as...",
            directory=str(self.folder_path.parent / "viewport.png"),
        )[0]
        if file_name == "":
            return
        p = Path(file_name)
        if p.suffix == "":
            file_name = str(p.with_suffix(".png"))
        pixmap = self.view.viewport().grab()
        pixmap.save(file_name)
