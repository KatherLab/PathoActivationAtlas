from PyQt5 import QtWidgets, QtGui, QtCore
from pathlib import Path


class Data(QtCore.QObject):
    def __init__(self, marker_file_name, sep_char="#", parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.save_dicts = None                              # dict of all save_dicts
        self.marker_file_name = Path(marker_file_name)      # path to e.g. "atlas" or "actgrid" file
        self.sep_char = sep_char                            # character used to separate key components in the overlay hierarchy
        self.exp_folder = None                              # path to the experiment folder, usually the one containing the marker file
        self.exp_name = None                                # Some identifying string for the currently opened file (used for the window title)
        self.creator_config = None                          # marker_file or .yaml config file opened
        self.layer_names = None                             # ya know
        self.thumbnail_pixmaps = None                       # dict with layer_name --> pixmap of thumbnail to use in the layer list
        self.extra_thumbnails = None                        # List of additional pixmaps to use in the thumbnail widget. Can be empty, will be same across all layers.
        self.tile_sizes = None    # dict(layer_name --> OrderedDict(div_level --> pixmap size for the tile images)) with order going from largest to smallest size
        self.tile_scales = None   # dict(layer_name --> dict((y,x) --> counts_scale_factor)
        self.tiles_scalable = None                          # True if anything other than ones in tile_scales, else False
        self.num_cells = None     # dict(layer_name --> number of grid cells in either direction)
        self.overlay_hierarchy = None                       # OverlayHierarchy object
        self.has_attributions = False                        # does this kind of visualisation come with attribution values?
        self.has_ground_truth = False                        # does this kind of visualisation come with extra ground truth values?
        self.has_metrics = False

    def set_data_status(self):
        if self.save_dicts is not None:
            self.has_attributions = ("avg_attributions_default" in self.save_dicts[self.layer_names[0]]
                                     or "attributions_default" in self.save_dicts[self.layer_names[0]])
            self.has_ground_truth = "extra_data" in self.save_dicts[self.layer_names[0]]
            self.has_metrics = "metrics" in self.save_dicts[self.layer_names[0]]

    def get_overlay_data(self, layer_name, overlay_id, **extra_params):
        # should return a torch.tensor of shape [num_cells, num_cells, 2], where in dimension 2:
        # - element 0 indicates a class index
        # - element 1 indicates a "class strength" of some sort (between 0 and 1)
        raise NotImplementedError()

    def get_overlay_labels(self, overlay_id):
        # should return a list of tuples where each tuple contains:
        # - element 0: class index (used for picking colors)
        # - element 1: class name
        # for the given overlay_id
        raise NotImplementedError()

    # dict(layer_name --> attribution in shape [num_cells, num_cells, num_classes])
    def raw_attributions_default(self):
        raise NotImplementedError()

    # rel attribution in shape [num_cells, num_cells, num_classes] --> must be updated
    def rel_attributions_default(self):
        raise NotImplementedError()

    # dict(layer_name --> col_name --> list of (cls_index, cls_name, rel_count, abs_count)
    # leaves may alternatively be just a single value that is not an instance of list.
    def cat_col_data(self):
        raise NotImplementedError()

    # dict(layer_name --> col_name --> (mean, std, median, min, max)-dict)
    # leaves may alternatively just be a single value that is not an instance of dict
    def cont_col_data(self):
        raise NotImplementedError()

    # dict(layer_name --> col_name --> list of (cls_index, cls_name, rel_count, abs_count)
    def metric_col_data(self):
        raise NotImplementedError()

    def update_rel_attributions(self, attr_scale_factor):
        """
        fills/updates the rel_attributions* entries - needs to be called when the attribution scale factor changes or is first set.
        """
        raise NotImplementedError()

class OverlayHierarchy:
    def __init__(self, hierarchy_dict, sep_char):
        self.dict = hierarchy_dict # (ordered?) dict with a structure like:
        """
        (name, txt) --> (name, txt) --> ... --> ((name, txt), (name, txt), ...)
                    --> (name, txt) --> ...
                    ...
        """
        # don't use sep_char in the names.
        self.sep_char = sep_char

    def create_menu(self, parent_menu=None, action_group=None, curr_dict=None, own_prefix=None):
        """
        returns:
         * a QMenu containing QMenus/QActions corresponding to the hierarchy, such that all options are mutually exclusive.
         (entries get their prefixed name as objectName)
         * the QActionGroup containing all the QActions (not the QMenus)
        """
        if parent_menu is None:
            parent_menu = QtWidgets.QMenu()
            curr_dict = self.dict
            own_prefix = ""
            action_group = QtWidgets.QActionGroup(parent_menu)
        for (name, txt) in curr_dict:
            new_prefix = f"{name}" if own_prefix == "" else f"{own_prefix}{self.sep_char}{name}"
            if isinstance(curr_dict[(name, txt)], dict):
                new_menu = QtWidgets.QMenu(parent_menu)
                new_menu.setObjectName(new_prefix)
                new_menu.setTitle(txt)
                self.create_menu(new_menu, action_group, curr_dict[(name, txt)], new_prefix)
                parent_menu.addMenu(new_menu)
            elif curr_dict[(name, txt)] is None:
                new_action = parent_menu.addAction(txt)
                new_action.setText(txt)
                new_action.setObjectName(new_prefix)
                new_action.setCheckable(True)
                if name == "disabled":
                    new_action.setChecked(True)
                action_group.addAction(new_action)
            else:
                new_menu = QtWidgets.QMenu(parent_menu)
                new_menu.setObjectName(new_prefix)
                new_menu.setTitle(txt)
                for (inner_name, inner_txt) in curr_dict[(name, txt)]:
                    new_new_prefix = f"{new_prefix}{self.sep_char}{inner_name}"
                    new_action = new_menu.addAction(inner_txt)
                    new_action.setText(inner_txt)
                    new_action.setObjectName(new_new_prefix)
                    new_action.setCheckable(True)
                    action_group.addAction(new_action)
                parent_menu.addMenu(new_menu)
        return parent_menu, action_group