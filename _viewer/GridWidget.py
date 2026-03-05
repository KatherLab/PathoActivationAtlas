from PyQt5 import QtWidgets, QtGui, QtCore
import cachetools
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

from _creator import utils
from .misc import *
from .GridLayerWidget import GridLayerWidget


class GridWidget(QtWidgets.QStackedWidget):
    def __init__(self, main_window, data, tile_cache_size="8 GB", overlay_cache_size="1 GB", parent=None, *super_args, **super_kwargs):
        super().__init__(parent, *super_args, **super_kwargs)
        self.main_window = main_window
        self.data = data
        self.data.setParent(self)
        self.main_window.setWindowTitle(f"{self.data.exp_name} | Overlay: disabled")
        self.current_layer = None
        self.options = {
            "scale_by_counts": False,
            "attribution_scale_factor": 10000,
            "overlay_opacity": 1.0,
            "active_overlay": "disabled",
            "thumbnail_mode": -1, # -1 == grid preview, numbers beyond that index into self.data.extra_thumbnails
            "tooltip_font_size": 9,
            "tooltip_num_cols": 13
        }

        # set up the entries in options
        # show next thumbnail
        if len(self.data.extra_thumbnails) > 0:
            thumbnail_btn = self.main_window.options.addAction(
                QtGui.QIcon(str(self.main_window.resource_dir/"forward-solid.svg")),
                "Show next thumbnail", self.update_thumbnail_mode
            )
        # Scale by counts
        if self.data.tiles_scalable:
            scale_by_counts_btn = self.main_window.options.addAction(
                QtGui.QIcon(str(self.main_window.resource_dir/"down-left-and-up-right-to-center-solid.svg")),
                "Scale by counts"
            )
            scale_by_counts_btn.setCheckable(True)
            self.options["scale_by_counts"] = scale_by_counts_btn.isChecked()
            scale_by_counts_btn.toggled.connect(self.toggle_scale_by_counts)
        # Tooltip font size
        tooltip_font_size_entry = IntLineEditWithTitle(
            "Tooltip font size: ", self.options["tooltip_font_size"], min_val=1
        )
        tooltip_font_size_entry.value_changed_signal.connect(self.update_tooltip_font_size)
        self.main_window.options.addWidget(tooltip_font_size_entry)
        # Max. number of data columns before line break in tooltip
        tooltip_num_cols_entry = IntLineEditWithTitle(
            "Tooltip max. #columns: ", self.options["tooltip_num_cols"], min_val=1
        )
        tooltip_num_cols_entry.value_changed_signal.connect(self.update_tooltip_num_cols)
        self.main_window.options.addWidget(tooltip_num_cols_entry)
        # Attribution factor
        attribution_factor_entry = DoubleLineEditWithTitle(
            "Attribution scale factor: ", self.options["attribution_scale_factor"]
        )
        attribution_factor_entry.value_changed_signal.connect(self.update_attribution_scale_factor)
        self.data.update_rel_attributions(self.options["attribution_scale_factor"])
        self.main_window.options.addWidget(attribution_factor_entry)
        # Overlay opacity
        opacity_slider = SliderWithTitle("Overlay opacity", QtCore.Qt.Horizontal, 0, 1)
        opacity_slider.value_changed_signal.connect(self.update_overlay_opacity)
        self.main_window.options.addWidget(opacity_slider)
        # Overlay menu
        overlay_btn = self.main_window.options.addAction(
            QtGui.QIcon(str(self.main_window.resource_dir/"layer-group-solid.svg")),
            "Overlay"
        )
        self.overlay_menu, self.overlay_action_group = self.data.overlay_hierarchy.create_menu()
        self.overlay_action_group.triggered.connect(self.update_overlay)
        overlay_btn.setMenu(self.overlay_menu)
        self.main_window.options.widgetForAction(overlay_btn).setPopupMode(QtWidgets.QToolButton.InstantPopup)
        # Viewport to image file
        viewport_to_image_btn = self.main_window.options.addAction(
            QtGui.QIcon(str(self.main_window.resource_dir/"crop-simple-solid.svg")),
            "Export viewport", self.viewport_to_image_file
        )
        # Scene to image file
        scene_to_image_btn = self.main_window.options.addAction(
            QtGui.QIcon(str(self.main_window.resource_dir/"panorama-solid.svg")),
            "Export scene", self.scene_to_image_file
        )

        # set up layer_list
        for layer in self.data.layer_names:
            entry = ListWidgetItem()
            entry.setIcon(QtGui.QIcon(self.data.thumbnail_pixmaps[layer]))
            entry.setText(layer)
            entry.setObjectName(f"{layer}_list_item")
            self.main_window.layer_list.addItem(entry)
        self.main_window.layer_list.currentItemChanged.connect(self.main_window.thumbnail_widget.switch_layer)
        self.main_window.layer_list.currentItemChanged.connect(self.switch_layer)

        # set up pixmap_cache
        div_lvls = self.data.creator_config["thumbnail_div_levels"]
        cache_factors = div_lvls[:]
        for i in range(len(cache_factors)):
            cache_factors[i] = cache_factors[i]**2 # space requirement decreases quadratically with scale factor
        cache_sizes = divide_cache_size_free_divisors(tile_cache_size, cache_factors)
        self.pixmap_cache = OrderedDict([
            ("orig", cachetools.LRUCache(maxsize=cache_sizes[0], getsizeof=get_size_of_pixmap))] + [
            (f"div_{lvl}", cachetools.LRUCache(maxsize=cache_sizes[i+1], getsizeof=get_size_of_pixmap)) for i, lvl in enumerate(div_lvls)
        ])

        # cache for storing data needed for drawing overlays (e.g. softmaxed attributions)
        # keys should be of format {layer}#{overlay_identifier} (where the identifier may include the attribution scale factor),
        # values should be torch.tensors with y,x as their first dimensions
        #print(overlay_cache_size.split())
        #print(size_to_bytes(overlay_cache_size))
        ocs_amount, ocs_unit = overlay_cache_size.split()
        ocs_amount = float(ocs_amount)
        self.overlay_cache = cachetools.LRUCache(
            maxsize=size_to_bytes(ocs_amount, ocs_unit), getsizeof=get_size_of_tensor
        )

        # pre-create pixmaps to use for overlays
        overlay_colors = [np.array(
            [c[0]*255, c[1]*255, c[2]*255], dtype=np.uint8
        ) for c in plt.cm.tab10.colors]
        white_color = np.array((255,255,255), dtype=np.uint8)
        num_colors = len(overlay_colors)
        overlay_pixels = np.zeros((num_colors, 1, 1, 3), dtype=np.uint8)
        for i, c in enumerate(overlay_colors):
            overlay_pixels[i,0,0] = c
        white_pixel = np.zeros((1,1,3), dtype=np.uint8)
        white_pixel[0,0] = white_color
        self.overlay_pixmaps = [QtGui.QPixmap.fromImage(QtGui.QImage(overlay_pixels[i], 1, 1, QtGui.QImage.Format_RGB888)) for i in range(num_colors)]
        self.white_pixmap = QtGui.QPixmap.fromImage(QtGui.QImage(white_pixel, 1, 1, QtGui.QImage.Format_RGB888))

        # set up the inner grid widgets
        self.layer_grids = dict()
        for layer_name in self.data.layer_names:
            self.layer_grids[layer_name] = GridLayerWidget(layer_name, self)
            self.addWidget(self.layer_grids[layer_name])
        null_grid = QtWidgets.QWidget(self)
        null_grid.setMinimumSize(QtCore.QSize(0,0))
        self.addWidget(null_grid)
        self.setCurrentWidget(null_grid)

        # connect viewport controls from the thumbnail widget
        self.main_window.thumbnail_widget.mouse_drag_signal.connect(self.update_relative_view_center)
        self.main_window.thumbnail_widget.mouse_scroll_signal.connect(self.update_view_scroll)

    ########## slots ##########
    def switch_layer(self, list_item, _):
        layer_name = list_item.objectName()[:-10] # cut off the "_list_item" suffix
        self.current_layer = layer_name
        self.setCurrentWidget(self.layer_grids[layer_name])
        self.refresh_views(False)

    def toggle_scale_by_counts(self, checked):
        self.options["scale_by_counts"] = checked
        self.refresh_views()

    def update_overlay_opacity(self, new_val):
        self.options["overlay_opacity"] = new_val
        self.refresh_views()

    def update_tooltip_font_size(self, new_val):
        self.options["tooltip_font_size"] = int(new_val)

    def update_tooltip_num_cols(self, new_val):
        self.options["tooltip_num_cols"] = int(new_val)

    def update_attribution_scale_factor(self, new_val):
        if (new_val != self.options["attribution_scale_factor"]):
            self.options["attribution_scale_factor"] = new_val
            self.data.update_rel_attributions(self.options["attribution_scale_factor"])
            self.refresh_views()

    def update_overlay(self, action):
        self.main_window.setWindowTitle(
            f"{self.data.exp_name} | Overlay: {action.objectName().replace(self.data.sep_char, ' >> ')}"
        )
        self.options["active_overlay"] = action.objectName()
        self.refresh_views()

    def get_overlay_cache_key(self):
        if self.options["active_overlay"] != "disabled":
            cache_key = f"{self.current_layer}{self.data.sep_char}{self.options['active_overlay']}"
            if self.options["active_overlay"].startswith("attribution"):
                attr_scale = self.options['attribution_scale_factor']
                cache_key = f"{cache_key}{self.data.sep_char}{attr_scale}"
            return cache_key
        else:
            return None

    def update_overlay_cache(self):
        if self.options["active_overlay"] != "disabled":
            cache_key = f"{self.current_layer}{self.data.sep_char}{self.options['active_overlay']}"
            extra_params = dict()
            if self.options["active_overlay"].startswith("attribution"):
                attr_scale = self.options['attribution_scale_factor']
                cache_key = f"{cache_key}{self.data.sep_char}{attr_scale}"
                extra_params["attribution_scale_factor"] = attr_scale
            if cache_key not in self.overlay_cache:
                self.overlay_cache[cache_key] = self.data.get_overlay_data(
                    self.current_layer, self.options["active_overlay"], **extra_params
                )
        cls_names_and_indices = self.data.get_overlay_labels(self.options["active_overlay"])
        class_name_to_pixmap = dict()
        for i, cls_name in cls_names_and_indices:
            class_name_to_pixmap[cls_name] = self.overlay_pixmaps[i%len(self.overlay_pixmaps)]
        self.main_window.legend.reset(class_name_to_pixmap)

    def update_thumbnail_mode(self):
        num_extra = len(self.data.extra_thumbnails)
        idx = self.options["thumbnail_mode"] + 1
        self.options["thumbnail_mode"] = idx if idx < num_extra else -1
        self.refresh_views()

    def refresh_views(self, repaint=True):
        if self.current_layer is not None:
            self.update_overlay_cache()
            if repaint:
                self.layer_grids[self.current_layer].scene.update()
            if self.options["thumbnail_mode"] == -1:
                scene_img = self.layer_grids[self.current_layer].get_scene_img()
                self.main_window.thumbnail_widget.update_base_pixmap(QtGui.QPixmap.fromImage(scene_img))
            else:
                self.main_window.thumbnail_widget.update_base_pixmap(
                    self.data.extra_thumbnails[self.options["thumbnail_mode"]]
                )

    def update_relative_view_center(self, rel_view_center):
        if self.current_layer is not None:
            scene_rect = self.layer_grids[self.current_layer].scene.sceneRect()
            scene_view_center = QtCore.QPointF(
                utils.map_to_range(rel_view_center.x(), 0.0, 1.0, scene_rect.left(), scene_rect.right()),
                utils.map_to_range(rel_view_center.y(), 0.0, 1.0, scene_rect.top(), scene_rect.bottom())
            )
            self.layer_grids[self.current_layer].view.centerOn(scene_view_center)

    def update_view_scroll(self, angle_delta):
        if self.current_layer is not None:
            self.layer_grids[self.current_layer].view._wheelEvent(angle_delta)

    def scene_to_image_file(self):
        if self.current_layer is not None:
            file_name = QtWidgets.QFileDialog.getSaveFileName(
                parent=self.main_window,
                caption="Save scene image as...",
                directory=str(self.data.exp_folder),
                filter="Image files (*.png *.jpg)"
            )[0]
            if file_name == "": return
            qimg = self.layer_grids[self.current_layer].get_scene_img()
            qimg.save(file_name)

    def viewport_to_image_file(self):
        if self.current_layer is not None:
            file_name = QtWidgets.QFileDialog.getSaveFileName(
                parent=self.main_window,
                caption="Save viewport image as...",
                directory=str(self.data.exp_folder),
                filter="Image files (*.png *.jpg)"
            )[0]
            if file_name == "": return
            pixmap = self.layer_grids[self.current_layer].view.viewport().grab()
            pixmap.save(file_name)