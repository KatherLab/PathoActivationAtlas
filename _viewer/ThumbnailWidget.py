from PyQt5 import QtWidgets, QtGui, QtCore


class ThumbnailWidget(QtWidgets.QWidget):
    mouse_drag_signal = QtCore.pyqtSignal("QPointF")
    mouse_scroll_signal = QtCore.pyqtSignal(int)
    def __init__(self, main_window, parent=None, *super_args, **super_kwargs):
        super().__init__(parent, *super_args, **super_kwargs)
        self.main_window = main_window
        self.current_layer = None
        self.relative_viewport = None
        self.pen = QtGui.QPen()
        self.pen.setColor(QtGui.QColor("red"))
        self.pen.setWidth(2)
        self.pixmap_base = None
        self.pixmap_marked = None

    def mouseMoveEvent(self, event:QtGui.QMouseEvent=None):
        self._mouseEvent(event)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event:QtGui.QMouseEvent=None):
        self._mouseEvent(event)
        super().mousePressEvent(event)

    def _mouseEvent(self, event:QtGui.QMouseEvent):
        buttons = event.buttons()
        if buttons == QtCore.Qt.LeftButton:
            pos = event.pos()
            relative_pos = QtCore.QPointF(pos.x() / self.width(), pos.y() / self.height())
            self.mouse_drag_signal.emit(relative_pos)

    def wheelEvent(self, event:QtGui.QWheelEvent):
        angle_delta = event.angleDelta().y()
        self.mouse_scroll_signal.emit(angle_delta)
        super().wheelEvent(event)

    def paintEvent(self, event:QtGui.QPaintEvent=None):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.eraseRect(self.rect())
        if self.current_layer is None: return
        self.pixmap_marked = self.pixmap_base.copy()
        self.pen.setWidth(max(self.pixmap_marked.width() // 120, 1))
        pixmap_rect = self.pixmap_marked.rect()
        pixmap_rect = QtCore.QRect(
            QtCore.QPoint(
                int(pixmap_rect.left() - self.relative_viewport.left()*(pixmap_rect.left()-pixmap_rect.right())),
                int(pixmap_rect.top() - self.relative_viewport.top()*(pixmap_rect.top()-pixmap_rect.bottom()))
            ),
            QtCore.QPoint(
                max(int(pixmap_rect.left() - self.relative_viewport.right()*(pixmap_rect.left()-pixmap_rect.right()))-1, 0),
                max(int(pixmap_rect.top() - self.relative_viewport.bottom()*(pixmap_rect.top()-pixmap_rect.bottom()))-1, 0)
            )
        )
        pixmap_painter = QtGui.QPainter(self.pixmap_marked)
        pixmap_painter.setPen(self.pen)
        pixmap_painter.drawRect(pixmap_rect)
        painter.drawPixmap(self.rect(), self.pixmap_marked)

    def heightForWidth(self, w):
        return w

    def hasHeightForWidth(self):
        return True

    ############### slots ##################
    def switch_layer(self, list_item, _):
        layer_name = list_item.objectName()[:-10] # cut off the "_list_item" suffix
        self.current_layer = layer_name
        self.main_window.thumbnail_area_layout.update()

    def receive_new_viewport(self, relative_rect):
        self.relative_viewport = relative_rect
        self.repaint()

    def update_base_pixmap(self, new_base):
        self.pixmap_base = new_base
        if self.pixmap_base.width() > 1000:
            self.pixmap_base = self.pixmap_base.scaledToWidth(1000)
