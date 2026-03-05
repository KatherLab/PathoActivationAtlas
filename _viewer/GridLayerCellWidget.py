from PyQt5 import QtWidgets, QtGui, QtCore
from _viewer.misc import get_table_cat, get_table_cont


class GridLayerCellWidget(QtWidgets.QWidget):
    def __init__(self, layer_name, pos, grid_widget, grid_layer_widget, parent=None, *super_args,
                 **super_kwargs):
        super().__init__(parent, *super_args, **super_kwargs)
        self.data = grid_widget.data
        self.grid_widget = grid_widget
        self.grid_layer_widget = grid_layer_widget
        self.layer_name = layer_name
        self.y = pos[0]
        self.x = pos[1]
        self.pos_str = f"{pos[0]}_{pos[1]}"
        self.id_str = f"{layer_name}_{self.pos_str}"
        self.size_scale_factor = self.data.tile_scales[self.layer_name][pos]
        self.setFixedSize(self.grid_layer_widget.cell_size)
        self.tooltip_font_size = -1
        self.tooltip_num_cols = -1

    def get_tooltip_data(self):
        tt_data = dict()
        if self.data.has_attributions:
            tt_data["metrics"] = dict()
            # contains (cls_idx, (raw_attr, rel_attr)) entries, sorted by raw_attr
            attr_default = sorted(list(enumerate(list(zip(
                self.data.raw_attributions_default()[self.layer_name][self.y, self.x], # shape (num_classes,)
                self.data.rel_attributions_default()[self.layer_name][self.y, self.x]
            )))), key=lambda x: x[1][0], reverse=True)
            # now ((cls_idx, cls_name), (raw_attr, rel_attr))
            attr_default = [((cls_idx, self.data.classes[cls_idx]), data) for cls_idx, data in attr_default]
            tt_data["metrics"]["Attributions"] = []
            for i, ((cls_idx, cls_name), (raw_attr, rel_attr)) in enumerate(attr_default):
                tt_data["metrics"]["Attributions"] += [{
                    "cls_name": cls_name,
                    "rel": rel_attr,
                    "abs": raw_attr
                }]

        if self.data.has_metrics:
            if "metrics" not in tt_data:
                tt_data["metrics"] = dict()
            metric_col_data = self.data.metric_col_data()[self.layer_name]
            for col_name in metric_col_data:
                low_best = metric_col_data[col_name]["low_best"]
                tile_data = sorted([(
                    cls_idx, cls_name, rel_count[self.y, self.x], abs_count[self.y, self.x]
                ) for (
                    cls_idx, cls_name, rel_count, abs_count
                ) in metric_col_data[col_name]["data"]], key=lambda x: x[3], reverse=not low_best)
                tt_data["metrics"][col_name] = []
                for i, (cls_idx, cls_name, rel_count, abs_count) in enumerate(tile_data):
                    tt_data["metrics"][col_name] += [{
                        "cls_name": cls_name,
                        "rel": rel_count,
                        "abs": abs_count
                    }]

        if self.data.has_ground_truth:
            cat_col_data = self.data.cat_col_data()[self.layer_name]
            if len(cat_col_data) != 0:
                tt_data["groundtruth_cat"] = dict()
                for col_name in cat_col_data:
                    if not isinstance(cat_col_data[col_name], dict): # actgrid case --> contains just a single value
                        tt_data["groundtruth_cat"][col_name] = cat_col_data[col_name]
                    else:
                        # atlas/class vis case --> dict with keys "data", "low_best"
                        # "data"-key contains list of (cls_idx, cls_name, rel_count, abs_count, low_best) tuples
                        low_best = cat_col_data[col_name]["low_best"]
                        tile_data = sorted([(
                            cls_idx, cls_name, rel_count[self.y, self.x], abs_count[self.y, self.x]
                        ) for (
                            cls_idx, cls_name, rel_count, abs_count
                        ) in cat_col_data[col_name]["data"]], key=lambda x: x[3], reverse=not low_best)
                        tt_data["groundtruth_cat"][col_name] = []
                        for i, (cls_idx, cls_name, rel_count, abs_count) in enumerate(tile_data):
                            if abs_count != 0:
                                tt_data["groundtruth_cat"][col_name] += [{
                                    "cls_name": cls_name,
                                    "rel": rel_count,
                                    "abs": abs_count
                                }]

            cont_col_data = self.data.cont_col_data()[self.layer_name]
            if len(cont_col_data) != 0:
                tt_data["groundtruth_cont"] = dict()
                for col_name in cont_col_data:
                    if not isinstance(cont_col_data[col_name], dict):  # actgrid case, contains just a single value
                        tt_data["groundtruth_cont"] = cont_col_data[col_name]
                    else:
                        col_data = cont_col_data[col_name]
                        tt_data["groundtruth_cont"][col_name] = {
                            "mean": col_data["mean"][self.y, self.x],
                            "std": col_data["stddev"][self.y, self.x],
                            "median": col_data["median"][self.y, self.x],
                            "min": col_data["min"][self.y, self.x],
                            "max": col_data["max"][self.y, self.x]
                        }
        return tt_data

    def build_tooltip(self):
        tooltip = "<p style='white-space:pre'>"
        tt_data = self.get_tooltip_data()
        if "metrics" in tt_data:
            table = get_table_cat(tt_data["metrics"], self.tooltip_num_cols)
            tooltip += f"<b><u>Attribution and other metrics:</b></u><br>{table}<br><br>"
        if "groundtruth_cat" in tt_data:
            table = get_table_cat(tt_data["groundtruth_cat"], self.tooltip_num_cols)
            tooltip += f"<b><u>Ground Truth (categorical):</b></u><br>{table}<br><br>"
        if "groundtruth_cont" in tt_data:
            table = get_table_cont(tt_data["groundtruth_cont"], self.tooltip_num_cols)
            tooltip += f"<b><u>Ground Truth (continuous):</b></u><br>{table}<br><br>"
        if tooltip != "<p style='white-space:pre'>":
            tooltip = tooltip[:-8] # cut off last linebreaks
        tooltip = tooltip.replace(
            "<th>", "<th style='padding-left:5px;text-align:left'>"
        ).replace(
            "<td>", "<td style='padding-left:5px;text-align:left'>"
        )
        tooltip += "</p>"
        return tooltip

    def event(self, event=None):
        if event.type() == QtCore.QEvent.ToolTip:
            tooltip_fs = self.grid_widget.options["tooltip_font_size"]
            tooltip_nc = self.grid_widget.options["tooltip_num_cols"]
            if self.toolTip() == "" or self.tooltip_font_size != tooltip_fs or self.tooltip_num_cols != tooltip_nc:
                self.tooltip_font_size = tooltip_fs
                self.tooltip_num_cols = tooltip_nc
                self.setStyleSheet(f"QToolTip {{font-size:{self.tooltip_font_size}px;}}")
                self.setToolTip(self.build_tooltip())
        return super().event(event)

    def paintEvent(self, QPaintEvent=None):
        # check for appropriate pyramid level: should be >= the current display_size
        # (gotta reconstruct the sizes bc they won't copy otherwise)
        cell_size = QtCore.QSize(
            self.grid_layer_widget.cell_size.width(), self.grid_layer_widget.cell_size.height()
        )
        display_size = QtCore.QSize(
            self.grid_layer_widget.displayed_cell_size.width(), self.grid_layer_widget.displayed_cell_size.height()
        )
        if display_size.isEmpty(): # don't bother with painting a pixmap if the displayed size is 0 anyways
            super().paintEvent(QPaintEvent)
            return
        # again, make sure to *COPY*
        real_size = QtCore.QSize(cell_size.width(), cell_size.height())
        if self.grid_widget.options["scale_by_counts"]:
            display_size *= self.size_scale_factor
            real_size *= self.size_scale_factor
        picked_lvl = "orig"
        for lvl, size in reversed(self.data.tile_sizes[self.layer_name].items()):
            if size.height() >= display_size.height():
                picked_lvl = lvl
                break
        # check if image is in the cache
        if self.id_str in self.grid_widget.pixmap_cache[picked_lvl]:
            pixmap = self.grid_widget.pixmap_cache[picked_lvl][self.id_str]
        else: # load into cache
            p = self.data.exp_folder/self.layer_name
            if picked_lvl != "orig":
                p = p/picked_lvl
            p = str(p/f"{self.pos_str}.png")
            pixmap = QtGui.QPixmap(p)
            self.grid_widget.pixmap_cache[picked_lvl][self.id_str] = pixmap
        # add overlay to the pixmap, if applicable
        cache_key = self.grid_widget.get_overlay_cache_key()
        if cache_key is not None:
            cls_idx, cls_magnitude = self.grid_widget.overlay_cache[cache_key][self.y, self.x]
            overlay_pixmap = self.grid_widget.overlay_pixmaps[int(cls_idx)%len(self.grid_widget.overlay_pixmaps)].scaled(pixmap.size())
            white_pixmap = self.grid_widget.white_pixmap.scaled(pixmap.size())
            pixmap = self.make_overlay(pixmap, overlay_pixmap, white_pixmap, cls_magnitude)
        # draw pixmap unto the cell widget
        padding = (cell_size - real_size) / 2
        target = QtCore.QRect(padding.width(), padding.height(), real_size.width(), real_size.height())
        painter = QtGui.QPainter(self)
        painter.drawPixmap(target, pixmap)
        self.setToolTip(self.build_tooltip())
        super().paintEvent(QPaintEvent)

    def make_overlay(self, orig_pixmap, overlay_pixmap, white_pixmap, cls_magnitude):
        opacity = self.grid_widget.options["overlay_opacity"]
        overlayed_pixmap1 = QtGui.QPixmap(orig_pixmap.size())
        overlayed_pixmap1.fill(QtCore.Qt.transparent)
        overlayed_pixmap2 = QtGui.QPixmap(orig_pixmap.size())
        overlayed_pixmap2.fill(QtCore.Qt.transparent)

        overlay_painter1 = QtGui.QPainter(overlayed_pixmap1)
        overlay_painter1.setOpacity(1.0-cls_magnitude)
        overlay_painter1.drawPixmap(0, 0, white_pixmap)
        overlay_painter1.setOpacity(cls_magnitude)
        overlay_painter1.drawPixmap(0, 0, overlay_pixmap)
        overlay_painter1.end()

        overlay_painter2 = QtGui.QPainter(overlayed_pixmap2)
        overlay_painter2.setOpacity(1.0-opacity)
        overlay_painter2.drawPixmap(0, 0, orig_pixmap)
        overlay_painter2.setOpacity(opacity)
        overlay_painter2.drawPixmap(0, 0, overlayed_pixmap1)
        overlay_painter2.end()

        return overlayed_pixmap2
