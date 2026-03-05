from PyQt5 import QtWidgets, QtGui, QtCore
import cachetools
from math import ceil
from typing import Tuple, Dict

from _annotator.GridCellWidget import GridCellWidget


class Legend(QtWidgets.QWidget):
    select_cls_idx_signal = QtCore.pyqtSignal(int)
    def __init__(
        self,
        class_info_to_pixmap: Dict[Tuple[int, str], QtGui.QPixmap] = None,
        parent=None,
        *super_args,
        **super_kwargs,
    ):
        super().__init__(parent, *super_args, **super_kwargs)
        self._layout = FlowLayout(self)
        if class_info_to_pixmap is None:
            class_info_to_pixmap = dict()
        self.buttons = QtWidgets.QButtonGroup(self)
        self.num_buttons = 0
        self.buttons.idClicked.connect(self.select_cls_idx)
        self.selected_cls_idx = -1
        self.reset(class_info_to_pixmap)

    def reset(self, class_info_to_pixmap=None):
        if class_info_to_pixmap is None:
            class_info_to_pixmap = dict()
        for button in self.buttons.buttons():
            self.buttons.removeButton(button)
            button.setParent(None)
            button.deleteLater()
        for child in self.children():
            if child.isWidgetType():
                self._layout.removeWidget(child)
                child.setParent(None)
                child.deleteLater()
        for (cls_idx, cls_name), pixmap in class_info_to_pixmap.items():
            color = QtWidgets.QLabel()
            color.setPixmap(pixmap)
            color.setScaledContents(True)
            color.setMinimumHeight(20)
            color.setMinimumWidth(20)
            txt = QtWidgets.QLabel()
            txt.setText(cls_name)
            legend_elem = QtWidgets.QPushButton()
            legend_elem.setCheckable(True)
            legend_elem_layout = QtWidgets.QHBoxLayout(legend_elem)
            legend_elem_layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
            legend_elem_layout.addWidget(color)
            legend_elem_layout.addWidget(txt)
            self._layout.addWidget(legend_elem)
            self.buttons.addButton(
                legend_elem, id=cls_idx if cls_idx != -1 else len(class_info_to_pixmap) - 1
            )
        if class_info_to_pixmap is not None:
            self.num_buttons = len(class_info_to_pixmap)
            if len(class_info_to_pixmap) != 0:
                self.buttons.button(self.num_buttons - 1).click()

    def select_cls_idx(self, clicked_id: int):
        if clicked_id == -1:
            raise ValueError("invalid button???? uh oh")
        self.select_cls_idx_signal.emit(clicked_id if clicked_id != self.num_buttons - 1 else -1)


# slightly modified from https://doc.qt.io/qtforpython-6/examples/example_widgets_layouts_flowlayout.html
class FlowLayout(QtWidgets.QLayout):
    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        self._item_list = []

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self._item_list.append(item)

    def count(self):
        return len(self._item_list)

    def itemAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list.pop(index)
        return None

    def expandingDirections(self):
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self._do_layout(QtCore.QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._item_list:
            size = size.expandedTo(item.minimumSize())
        size += QtCore.QSize(2 * self.contentsMargins().top(), 2 * self.contentsMargins().top())
        return size

    def _do_layout(self, rect, test_only):
        x = rect.x()
        y = rect.y()
        line_height = 0
        spacing = self.spacing()
        for item in self._item_list:
            style = item.widget().style()
            layout_spacing_x = style.layoutSpacing(
                QtWidgets.QSizePolicy.PushButton, QtWidgets.QSizePolicy.PushButton, QtCore.Qt.Horizontal
            )
            layout_spacing_y = style.layoutSpacing(
                QtWidgets.QSizePolicy.PushButton, QtWidgets.QSizePolicy.PushButton, QtCore.Qt.Vertical
            )
            space_x = spacing + layout_spacing_x
            space_y = spacing + layout_spacing_y
            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0
            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint()))
            x = next_x
            line_height = max(line_height, item.sizeHint().height())
        return y + line_height - rect.y()


class DoubleLineEditWithTitle(QtWidgets.QWidget):
    value_changed_signal = QtCore.pyqtSignal(float)

    def __init__(self, text, init_value, parent=None, *super_args, **super_kwargs):
        super().__init__(parent, *super_args, **super_kwargs)
        size_policy = self.sizePolicy()
        size_policy.setHorizontalPolicy(QtWidgets.QSizePolicy.MinimumExpanding)
        self.setSizePolicy(size_policy)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 1, 0, 1)
        self.setLayout(layout)
        self.validator = QtGui.QDoubleValidator(self)
        locale = QtCore.QLocale(QtCore.QLocale.C)
        locale.setNumberOptions(QtCore.QLocale.RejectGroupSeparator)
        self.validator.setLocale(locale)
        self.line_edit = QtWidgets.QLineEdit()
        self.line_edit.setValidator(self.validator)
        self.line_edit.setText(f"{init_value}")
        self.line_edit.editingFinished.connect(self.on_editing_finished)
        self.txt = QtWidgets.QLabel(text)
        layout.addWidget(self.txt)
        layout.addWidget(self.line_edit)

    def on_editing_finished(self):
        new_val = float(self.line_edit.text())
        self.value_changed_signal.emit(new_val)


class IntLineEditWithTitle(QtWidgets.QWidget):
    value_changed_signal = QtCore.pyqtSignal(int)

    def __init__(
        self, text, init_value, min_val=None, max_val=None, parent=None, *super_args, **super_kwargs
    ):
        super().__init__(parent, *super_args, **super_kwargs)
        size_policy = self.sizePolicy()
        size_policy.setHorizontalPolicy(QtWidgets.QSizePolicy.MinimumExpanding)
        self.setSizePolicy(size_policy)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 1, 0, 1)
        self.setLayout(layout)
        self.validator = QtGui.QIntValidator(self)
        if min_val is not None:
            self.validator.setBottom(min_val)
        if max_val is not None:
            self.validator.setTop(max_val)
        locale = QtCore.QLocale(QtCore.QLocale.C)
        locale.setNumberOptions(QtCore.QLocale.RejectGroupSeparator)
        self.validator.setLocale(locale)
        self.line_edit = QtWidgets.QLineEdit()
        self.line_edit.setValidator(self.validator)
        self.line_edit.setText(f"{init_value}")
        self.line_edit.editingFinished.connect(self.on_editing_finished)
        self.txt = QtWidgets.QLabel(text)
        layout.addWidget(self.txt)
        layout.addWidget(self.line_edit)

    def on_editing_finished(self):
        new_val = int(self.line_edit.text())
        self.value_changed_signal.emit(new_val)


class SliderWithTitle(QtWidgets.QWidget):
    value_changed_signal = QtCore.pyqtSignal(float)

    def __init__(self, text, orientation, min, max, parent=None, *super_args, **super_kwargs):
        super().__init__(parent, *super_args, **super_kwargs)
        size_policy = self.sizePolicy()
        size_policy.setHorizontalPolicy(QtWidgets.QSizePolicy.MinimumExpanding)
        self.setSizePolicy(size_policy)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 1, 0, 1)
        self.setLayout(layout)

        self.slider = QtWidgets.QSlider(orientation)
        self.slider.setTickInterval(100)
        self.slider.setMinimum(min)
        self.slider.setMaximum(max * self.slider.tickInterval())
        self.slider.setSingleStep(1)
        self.slider.setPageStep(self.slider.tickInterval() // 10)
        self.slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.slider.setValue(self.slider.maximum() * self.slider.tickInterval())
        self.base_txt = text
        start_value = round(float(self.slider.value() / self.slider.tickInterval()), 2)
        self.txt = QtWidgets.QLabel(f"{text} ({start_value:0.2f})")
        self.slider.valueChanged.connect(self.on_changed_value)
        layout.addWidget(self.txt)
        layout.addWidget(self.slider)

    def on_changed_value(self, value):
        new_val = round(float(value / self.slider.tickInterval()), 2)
        self.txt.setText(f"{self.base_txt} ({new_val:0.2f})")
        self.value_changed_signal.emit(new_val)


class ListWidgetItem(QtWidgets.QListWidgetItem):
    """QListWidgetItem with objectName"""

    def __init__(self, object_name: str = None, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        self.object_name = object_name

    def objectName(self):
        return self.object_name

    def setObjectName(self, new_object_name: str):
        self.object_name = new_object_name


class InteractiveQGraphicsView(QtWidgets.QGraphicsView):
    zoom_signal = QtCore.pyqtSignal(float)
    paint_event_signal = QtCore.pyqtSignal()
    # this signal should contain the current viewport area in scene coordinates
    viewport_to_scene_signal = QtCore.pyqtSignal("QRectF")
    right_btn_down_move_signal = QtCore.pyqtSignal("QPointF")

    # modified from:
    # https://stackoverflow.com/questions/19113532/qgraphicsview-zooming-in-and-out-under-mouse-position-using-mouse-wheel/29026916#29026916
    def __init__(self, parent=None):
        super(InteractiveQGraphicsView, self).__init__(parent)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.zoom_factor = 1.0
        self.zoom_in_factor = 1.1
        self.zoom_out_factor = 1 / self.zoom_in_factor

    def getSceneRect(self):
        tl_corner = self.mapToScene(QtCore.QPoint(0, 0))
        viewport_size = self.viewport().size()
        br_corner = self.mapToScene(QtCore.QPoint(viewport_size.width(), viewport_size.height()))
        return QtCore.QRectF(tl_corner, br_corner)

    def paintEvent(self, QEvent=None):
        super().paintEvent(QEvent)
        self.viewport_to_scene_signal.emit(self.getSceneRect())
        self.paint_event_signal.emit()

    def wheelEvent(self, event):
        self._wheelEvent(event.angleDelta().y(), event.pos())

    def mouseMoveEvent(self, QMouseEvent=None):
        if QMouseEvent.buttons() & QtCore.Qt.RightButton:
            scene_coords = self.mapToScene(QMouseEvent.pos())
            scene_rect = self.scene().sceneRect()
            if scene_rect.contains(scene_coords):
                self.right_btn_down_move_signal.emit(scene_coords)
        return super().mouseMoveEvent(QMouseEvent)

    def _wheelEvent(self, angle_delta, view_pos=None):
        if view_pos is None:
            view_pos = self.size() / 2
            view_pos = QtCore.QPoint(view_pos.width(), view_pos.height())
        # Save the scene pos
        oldPos = self.mapToScene(view_pos)
        # Zoom
        if angle_delta > 0:
            self.scale(self.zoom_in_factor, self.zoom_in_factor)
            self.zoom_factor *= self.zoom_in_factor
        else:
            self.scale(self.zoom_out_factor, self.zoom_out_factor)
            self.zoom_factor *= self.zoom_out_factor
        # Get the new position
        newPos = self.mapToScene(view_pos)
        # Move scene to old position
        delta = newPos - oldPos
        self.translate(delta.x(), delta.y())
        self.zoom_signal.emit(self.zoom_factor)


def get_size_of_pixmap(pixmap: QtGui.QPixmap):
    """
    Returns the (rough) size of a QPixmap in bytes
    """
    return int((pixmap.height() * pixmap.width() * pixmap.depth()) / 8)


def size_to_bytes(amount, unit):
    """
    Convert sizes given as {byte, KB, MB, GB, TB or PB} to byte. (Multipliers of 1024, not 1000, increase the unit size)
    """
    unit_mapping = {"byte": 1, "KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30, "TB": 1 << 40, "PB": 1 << 50}
    factor = unit_mapping[unit]
    return amount * factor


def divide_cache_size(size_str, num_caches, div_factor):
    """
    Divide a cache of size size_str (given as f"{number} {unit}") among num_caches caches such that each successive
    cache's size is divided by div_factor. Sizes returned are in bytes.
    """
    amount, unit = size_str.split(" ")
    amount = int(amount)
    bytes = size_to_bytes(amount, unit)
    sizes = [0 for _ in range(num_caches)]
    divisor = 0
    for i in range(num_caches):
        divisor += 1 / (div_factor**i)
    sizes[0] = int(bytes / divisor)
    for i in range(1, num_caches):
        sizes[i] = int(sizes[0] / (div_factor**i))
    return sizes


def divide_cache_size_free_divisors(size_str, div_factors):
    """
    Divide a cache of size size_str (given as f"{number} {unit}") among num_caches caches such that each caches' size
    is divided by div_factors[i]. No entry in div_factors needed for the very first cache. Sizes returned are in bytes.
    """
    amount, unit = size_str.split(" ")
    amount = int(amount)
    bytes = size_to_bytes(amount, unit)
    num_caches = len(div_factors) + 1
    sizes = [0 for _ in range(num_caches)]
    divisor = 1
    for i in range(1, num_caches):
        divisor += 1 / (div_factors[i - 1])
    sizes[0] = int(bytes / divisor)
    for i in range(1, num_caches):
        sizes[i] = int(sizes[0] / (div_factors[i - 1]))
    return sizes

def map_to_range(val, old_min, old_max, new_min, new_max):
    return ((val - old_min) * ((new_max - new_min) / (old_max - old_min))) + new_min