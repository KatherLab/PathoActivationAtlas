from PyQt5 import QtWidgets, QtGui, QtCore
import torch
from collections import OrderedDict
from natsort import natsorted

from _creator import utils
from .Data import *


class DataClassVis(Data):
    def __init__(self, marker_file_name, sep_char="#"):
        super().__init__(marker_file_name, sep_char)
        self.exp_folder = self.marker_file_name.parent
        self.exp_name = f"Class Visualization: {self.exp_folder.parent.parent.name} ({self.exp_folder.name})"
        _, self.creator_config = utils.open_config(self.exp_folder/"class_vis", "class_vis", mkdir=False)
        # layer names here means class names, I didn't really have this scenario in mind when writing the Data interface but it's whatever
        self.layer_names = natsorted([p.stem[5:] for p in self.exp_folder.iterdir() if p.suffix == ".pt"])

        self.thumbnail_pixmaps = dict()
        for layer in self.layer_names:
            p = self.exp_folder / f"{layer}_small.png"
            if not p.exists():
                print(f"[DataClassVis]: {p} not found.")
            self.thumbnail_pixmaps[layer] = QtGui.QPixmap(str(p))
        self.extra_thumbnails = []

        self.save_dicts = dict()
        self.num_cells = dict()
        for layer in self.layer_names:
            d = torch.load(self.exp_folder/f"grid_{layer}.pt")
            d = {
                k: d[k] for k in d.keys() if k in ["num_cells", "extra_data", "metrics"]
            }
            self.save_dicts[layer] = d
            self.num_cells[layer] = d["num_cells"]

        div_lvls = self.creator_config["thumbnail_div_levels"]
        self.tile_sizes = dict()
        for layer in self.layer_names:
            sizes = OrderedDict()
            img_path = str(next(p for p in (self.exp_folder/layer).iterdir() if p.suffix==".png"))
            sizes["orig"] = QtGui.QPixmap(img_path).size()
            for lvl in div_lvls:
                img_path = str(next(p for p in (self.exp_folder/layer/f"div_{lvl}").iterdir() if p.suffix==".png"))
                sizes[f"div_{lvl}"] = QtGui.QPixmap(img_path).size()
            self.tile_sizes[layer] = sizes

        self.tile_scales = {
            layer: {
                (y, x): 1 for y in range(self.num_cells[layer]) for x in range(self.num_cells[layer])
            } for layer in self.layer_names
        }
        self.tiles_scalable = False

        extra_cols = dict()
        for k in ["extra_data", "metrics"]:
            extra_cols[k] = self.save_dicts[self.layer_names[0]].get(k, dict()).keys()
            extra_cols[k] = tuple((
                col_name,
                "counts" in self.save_dicts[self.layer_names[0]].get(k, dict())[col_name]
            ) for col_name in extra_cols[k])

        self.cat_col_classes = dict()
        for (col_name, is_categorical) in extra_cols["extra_data"]:
            if not is_categorical: continue
            cls_set = set()
            for save_dict in self.save_dicts.values():
                cls_set.update(save_dict["extra_data"][col_name]["classes"])
            self.cat_col_classes[col_name] = sorted(list(cls_set))

        self.metric_col_classes = dict()
        for (col_name, is_categorical) in extra_cols["metrics"]:
            cls_set = set()
            for save_dict in self.save_dicts.values():
                cls_set.update(save_dict["metrics"][col_name]["classes"])
            self.metric_col_classes[col_name] = sorted(list(cls_set))

        hierarchy_dict = {
            ("disabled", "Disabled"): None,
            ("groundtruth", "Ground truth"): {
                (col_name, col_name): (
                    ("all", "All classes"),
                    *[(cls, cls) for cls in self.cat_col_classes[col_name]]
                ) if is_categorical else None for (col_name, is_categorical) in extra_cols["extra_data"]
            },
            ("metrics", "Metrics"): {
                (col_name, col_name): (
                    ("all", "All classes"),
                    *[(cls, cls) for cls in self.metric_col_classes[col_name]]
                ) for (col_name, is_categorical) in extra_cols["metrics"]
            }
        }
        self.overlay_hierarchy = OverlayHierarchy(hierarchy_dict, self.sep_char)
        self.prepare_getters()
        self.set_data_status()

    def get_overlay_data(self, layer_name, overlay_id, **extra_params):
        result = torch.zeros((self.num_cells[layer_name], self.num_cells[layer_name], 2))
        overlay_id_parts = overlay_id.split(self.sep_char)

        if overlay_id.startswith(("groundtruth", "metrics")):
            col_name = overlay_id_parts[1]
            extra_data = self.save_dicts[layer_name]["extra_data" if overlay_id.startswith("groundtruth") else "metrics"][col_name]
            if len(overlay_id_parts) == 3:  # == column contained categorical data
                low_best = extra_data.get("low_best", False)
                cls_name = overlay_id_parts[2]
                classes = extra_data["classes"]
                counts = extra_data["counts"]  # [num_cells, num_cells, len(classes)]
                total_counts_per_cell = torch.sum(counts.to(torch.float32), dim=2, keepdim=True)
                total_counts_per_cell[total_counts_per_cell == 0] = 1  # prevent divide by zero
                rel_counts = (counts.to(torch.float32) / total_counts_per_cell).squeeze(dim=2)
                if low_best:
                    rel_counts = 1.0 - rel_counts
                if cls_name == "all":
                    if low_best:
                        max_idx = torch.argmin(counts, dim=2, keepdim=True)
                    else:
                        max_idx = torch.argmax(counts, dim=2, keepdim=True)
                    rel_counts = torch.gather(rel_counts, dim=2, index=max_idx).squeeze(dim=2)
                    max_idx = max_idx.squeeze(dim=2)
                    result[:, :, 0] = max_idx
                    result[:, :, 1] = rel_counts
                else:
                    cls_idx = classes.index(cls_name)
                    result[:, :, 0] = cls_idx
                    result[:, :, 1] = rel_counts[:, :, cls_idx]
            else:  # column contained continuous data
                means = extra_data["mean"]  # [num_cells, num_cells]
                min_mean = torch.min(means[~means.isinf()])
                max_mean = torch.max(means[~means.isinf()])
                rel_means = utils.map_to_range(
                    means, old_min=min_mean, old_max=max_mean, new_min=0.0, new_max=1.0
                )
                result[:, :, 0] = 0
                result[:, :, 1] = rel_means
        else:
            raise ValueError(f"[DataClassVis::get_overlay_data]: Invalid overlay_id: {overlay_id}")
        return result


    def get_overlay_labels(self, overlay_id):
        overlay_id_parts = overlay_id.split(self.sep_char)
        if overlay_id.startswith(("groundtruth", "metrics")):
            col_name = overlay_id_parts[1]
            col_classes = self.cat_col_classes if overlay_id.startswith("groundtruth") else self.metric_col_classes
            if len(overlay_id_parts) == 3:  # == column contained categorical data
                cls_name = overlay_id_parts[2]
                if cls_name == "all":
                    return list(enumerate(col_classes[col_name]))
                else:
                    cls_idx = col_classes[col_name].index(cls_name)
                    return [(cls_idx, cls_name)]
            else:  # column contained continuous data
                return [(0, col_name)]
        else:
            return []


    def prepare_getters(self):
        self._cat_col_data = dict()
        self._cont_col_data = dict()
        self._metric_col_data = dict()
        for layer_name, save_dict in self.save_dicts.items():
            self._cat_col_data[layer_name] = dict()
            self._cont_col_data[layer_name] = dict()
            self._metric_col_data[layer_name] = dict()

            metric_cols = save_dict.get("metrics", dict()).keys()
            for col_name in metric_cols:
                data = save_dict["metrics"][col_name]
                self._metric_col_data[layer_name][col_name] = {
                    "data": [],
                    "low_best": data.get("low_best", False)
                }
                total_count = data["counts"].sum(dim=2)
                for cls_idx, cls_name in enumerate(data["classes"]):
                    total_count_nice = total_count
                    total_count_nice[total_count_nice == 0] = 1  # prevent divide by 0
                    rel_count = data["counts"][:, :, cls_idx] / total_count_nice
                    self._metric_col_data[layer_name][col_name]["data"] += [
                        (cls_idx, cls_name, rel_count, data["counts"][:, :, cls_idx])
                    ]

            extra_cols = save_dict.get("extra_data", dict()).keys()
            extra_cols = tuple((
                col_name,
                "counts" in save_dict["extra_data"][col_name]
            ) for col_name in extra_cols)

            for col_name, is_cat in extra_cols:
                extra_data = save_dict["extra_data"][col_name]

                if is_cat:
                    self._cat_col_data[layer_name][col_name] = {
                        "data": [],
                        "low_best": extra_data.get("low_best", False)
                    }
                    total_count = extra_data["counts"].sum(dim=2)
                    for cls_idx, cls_name in enumerate(extra_data["classes"]):
                        total_count_nice = total_count
                        total_count_nice[total_count_nice == 0] = 1  # prevent divide by 0
                        rel_count = extra_data["counts"][:, :, cls_idx] / total_count_nice
                        self._cat_col_data[layer_name][col_name]["data"] += [
                            (cls_idx, cls_name, rel_count, extra_data["counts"][:, :, cls_idx])
                        ]

                else:
                    self._cont_col_data[layer_name][col_name] = extra_data


    def cat_col_data(self):
        return self._cat_col_data

    def cont_col_data(self):
        return self._cont_col_data

    def metric_col_data(self):
        return self._metric_col_data

    def update_rel_attributions(self, attr_scale_factor):
        return None