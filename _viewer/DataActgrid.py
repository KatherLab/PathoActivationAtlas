from PyQt5 import QtWidgets, QtGui, QtCore
import torch
from torchvision import transforms as tf
import numpy as np
from collections import OrderedDict
from natsort import natsorted

from .Data import *
from _creator import utils

class DataActgrid(Data):
    def __init__(self, marker_file_name, sep_char="#"):
        super().__init__(marker_file_name, sep_char)
        self.exp_folder = self.marker_file_name.parent # folder for an image in this case
        img_name = self.exp_folder.stem
        self.exp_name = f"Actgrid: {img_name} ({self.exp_folder.parent.name})"
        _, self.creator_config = utils.open_config(self.exp_folder/"actgrid", "actgrid", mkdir=False)
        self.layer_names = natsorted([p.stem[5:] for p in self.exp_folder.iterdir() if p.suffix == ".pt"])  # cut off the grid_ prefix

        self.thumbnail_pixmaps = dict()
        for layer in self.layer_names:
            p = self.exp_folder/f"{layer}_small.png"
            if not p.exists():
                print(f"[DataActgrid]: {p} not found.")
            self.thumbnail_pixmaps[layer] = QtGui.QPixmap(str(p))
        input_img_path = [p for p in self.exp_folder.iterdir() if p.stem == "input_image" and not p.is_dir()][0]
        self.extra_thumbnails = [QtGui.QPixmap(str(input_img_path))]

        self.save_dicts = dict()
        self.num_cells = dict()
        for layer in self.layer_names:
            d = torch.load(self.exp_folder/f"grid_{layer}.pt")
            d = {
                k: d[k] for k in d.keys() if k in ["attributions_default", "num_cells", "extra_data", "img_idx"]
            }
            self.save_dicts[layer] = d
            self.num_cells[layer] = d["num_cells"]
            self.img_idx = d["img_idx"]

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

        self.classes = self.creator_config["class_names"]
        hierarchy_dict = {
            ("disabled", "Disabled"): None,
            ("attributiondefault", "Attribution"): (
                ("all", "All classes"),
                *[(cls, cls) for cls in self.classes]
            ),
        }
        self.overlay_hierarchy = OverlayHierarchy(hierarchy_dict, self.sep_char)
        self.prepare_getters()
        self.set_data_status()


    def get_overlay_data(self, layer_name, overlay_id, **extra_params):
        result = torch.zeros((self.num_cells[layer_name], self.num_cells[layer_name], 2))
        overlay_id_parts = overlay_id.split(self.sep_char)
        if not overlay_id.startswith("attribution"):
            raise ValueError(f"[DataActgrid::get_overlay_data]: Invalid overlay_id: {overlay_id}")
        attr_scale_factor = extra_params["attribution_scale_factor"]
        #raw_attr = self.save_dicts[layer_name][dict_key] # shape [num_classes, num_cells, num_cells]
        #raw_attr = raw_attr.permute((1,2,0)) # now [num_classes, num_classes, num_cells]
        raw_attr = self.raw_attributions_default()[layer_name]
        cls_name = overlay_id_parts[1]
        softmaxed_attr = torch.nn.functional.softmax(attr_scale_factor*raw_attr, dim=2)
        if cls_name == "all":
            max_cls = torch.argmax(raw_attr, dim=2, keepdim=True)
            softmaxed_attr = torch.gather(softmaxed_attr, dim=2, index=max_cls).squeeze(dim=2)
            max_cls = max_cls.squeeze(dim=2)
            softmaxed_attr = utils.map_to_range(
                softmaxed_attr,
                old_min = 1/len(self.classes), old_max=1.0,
                new_min=0.5, new_max=1.0
            )
            result[:,:,0] = max_cls
            result[:,:,1] = softmaxed_attr
        else:
            cls_idx = self.classes.index(cls_name)
            softmaxed_attr = softmaxed_attr[:,:,cls_idx]
            result[:,:,0] = cls_idx
            result[:,:,1] = softmaxed_attr
        return result


    def get_overlay_labels(self, overlay_id):
        overlay_id_parts = overlay_id.split(self.sep_char)
        if overlay_id.startswith("attribution"):
            cls_name = overlay_id_parts[1]
            if cls_name == "all":
                return list(enumerate(self.classes))
            else:
                cls_idx = self.classes.index(cls_name)
                return [(cls_idx, cls_name)]
        else:
            return []

    def prepare_getters(self):
        self._raw_attributions_default = dict()
        self._cat_col_data = dict()
        self._cont_col_data = dict()
        for layer_name, save_dict in self.save_dicts.items():
            self._raw_attributions_default[layer_name] = save_dict["attributions_default"].permute((1,2,0))
            self._cat_col_data[layer_name] = dict()
            self._cont_col_data[layer_name] = dict()
            for col_name in self.creator_config["extra_data_columns"]["categorical"]:
                self._cat_col_data[layer_name][col_name] = save_dict["extra_data"][col_name]
            for col_name in self.creator_config["extra_data_columns"]["continuous"]:
                self._cont_col_data[layer_name][col_name] = save_dict["extra_data"][col_name]

    # dict(layer_name --> attribution in shape [num_cells, num_cells, num_classes])
    def raw_attributions_default(self):
        return self._raw_attributions_default

    # rel attribution in shape [num_cells, num_cells, num_classes] --> must be updated
    def rel_attributions_default(self):
        return self._rel_attributions_default

    def cat_col_data(self):
        return self._cat_col_data

    def cont_col_data(self):
        return self._cont_col_data

    def update_rel_attributions(self, attr_scale_factor):
        """
        fills/updates the rel_attributions* entries - needs to be called when the attribution scale factor changes or is first set.
        """
        self._rel_attributions_default = dict()
        for layer_name in self.layer_names:
            self._rel_attributions_default[layer_name] = torch.nn.functional.softmax(
                attr_scale_factor*self.raw_attributions_default()[layer_name], dim=2
            )