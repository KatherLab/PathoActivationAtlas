from __future__ import annotations
from PyQt5 import QtWidgets, QtGui, QtCore
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from _annotator.GridWidget import GridWidget


class GridCellWidget(QtWidgets.QWidget):
    def __init__(
        self, grid_widget: GridWidget, pos: Tuple[int, int], init_cls: int = -1, *super_args, **super_kwargs
    ):
        super().__init__(grid_widget, *super_args, **super_kwargs)
        self.grid_widget = grid_widget
        self.y = pos[0]
        self.x = pos[1]
        self.pos_str = f"{pos[0]}_{pos[1]}"
        self.setFixedSize(self.grid_widget.tile_sizes["orig"])
        self.annotated_cls = init_cls

    def send_annotated_cls(self):
        self.grid_widget.annotation_data[(self.y, self.x)] = self.annotated_cls

    def receive_annotated_cls(self):
        old_cls = self.annotated_cls
        self.annotated_cls = self.grid_widget.annotation_data.get((self.y, self.x), self.annotated_cls)
        if old_cls != self.annotated_cls:
            self.update()

    def build_tooltip(self):
        if self.annotated_cls == -1:
            cls_name = "???"
        else:
            cls_name = self.grid_widget.class_names[self.annotated_cls]
        return f"({self.y}, {self.x}): {cls_name}"

    def update_annotated_cls(self):
        new_cls = self.grid_widget.options["cls_idx"]
        old_cls = self.annotated_cls
        if old_cls != new_cls:
            self.annotated_cls = new_cls
            self.update()
            self.grid_widget.refresh_views(False) # update the thumbnail

    def mousePressEvent(self, QMouseEvent=None):
        if QMouseEvent.button() == QtCore.Qt.RightButton:
            self.update_annotated_cls()
        return super().mousePressEvent(QMouseEvent)

    def event(self, event=None):
        if event.type() == QtCore.QEvent.ToolTip:
            self.setToolTip(self.build_tooltip())
        return super().event(event)

    def paintEvent(self, QPaintEvent=None):
        # check for appropriate pyramid level: should be >= the current display_size
        cell_size = QtCore.QSize(
            self.grid_widget.tile_sizes["orig"].width(), self.grid_widget.tile_sizes["orig"].height()
        )
        display_size = QtCore.QSize(
            self.grid_widget.displayed_cell_size.width(), self.grid_widget.displayed_cell_size.height()
        )
        if display_size.isEmpty():  # don't bother with painting a pixmap if the displayed size is 0 anyways
            return
        real_size = QtCore.QSize(cell_size.width(), cell_size.height())
        picked_lvl = "orig"
        for lvl, size in reversed(self.grid_widget.tile_sizes.items()):
            if size.height() >= display_size.height():
                picked_lvl = lvl
                break
                # check if image is in the cache
        if self.pos_str in self.grid_widget.pixmap_cache[picked_lvl]:
            pixmap = self.grid_widget.pixmap_cache[picked_lvl][self.pos_str]
        else:  # load into cache
            p = self.grid_widget.folder_path
            if picked_lvl != "orig":
                p = p / picked_lvl
            p = str(p / f"{self.pos_str}.png")
            pixmap = QtGui.QPixmap(p)
            self.grid_widget.pixmap_cache[picked_lvl][self.pos_str] = pixmap
        if self.annotated_cls != -1 and self.grid_widget.options["overlay_active"]:
            overlay_pixmap = self.grid_widget.overlay_pixmaps[
                self.annotated_cls % len(self.grid_widget.overlay_pixmaps)
            ].scaled(pixmap.size())
            pixmap = self.make_overlay(pixmap, overlay_pixmap)
        # draw pixmap unto the cell widget
        padding = (cell_size - real_size) / 2
        target = QtCore.QRect(padding.width(), padding.height(), real_size.width(), real_size.height())
        painter = QtGui.QPainter(self)
        painter.drawPixmap(target, pixmap)
        super().paintEvent(QPaintEvent)

    def make_overlay(self, orig_pixmap, overlay_pixmap):
        opacity = self.grid_widget.options["overlay_opacity"]
        overlayed_pixmap = QtGui.QPixmap(orig_pixmap.size())
        overlayed_pixmap.fill(QtCore.Qt.transparent)
        overlay_painter = QtGui.QPainter(overlayed_pixmap)
        overlay_painter.setOpacity(1.0 - opacity)
        overlay_painter.drawPixmap(0, 0, orig_pixmap)
        overlay_painter.setOpacity(opacity)
        overlay_painter.drawPixmap(0, 0, overlay_pixmap)
        overlay_painter.end()
        return overlayed_pixmap
