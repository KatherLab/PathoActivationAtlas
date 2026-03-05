from pathlib import Path
import torch
import yaml
from pydoc import locate
import torchvision.transforms as tf
import datetime
import shutil
import numpy as np
import functools
from collections import OrderedDict

from _creator import captum_fragments


def open_config(config_path, vis_type, mkdir=True):
    """
    - Open and return the configuration at config_path.
    - Create save folder
    - Figure out which device to use
    """
    with Path(config_path).open() as f:
        config = yaml.load(f, yaml.CSafeLoader)
    config["vis_type"] = vis_type
    config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    for layer_name in config["layers"]:
        if config["layers"][layer_name] is not None:
            config["layers"][layer_name] = tuple(
                [entry if isinstance(entry, tuple) else entry for entry in config["layers"][layer_name]]
            )
    save_root = Path(config["save_root"]) / config["name"] / vis_type
    if mkdir:
        save_root.mkdir(exist_ok=True, parents=True)
    return save_root, config


def load_checkpoint(model, checkpoint_path, param_key, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint[param_key], strict=True)
    return model


def locate_and_init(cls_name, prefixes, args):
    cls = None
    for prefix in prefixes:
        if cls is None:
            cls = locate(prefix+cls_name)
    if cls is None:
        raise ImportError(f"{cls_name} not found in {prefixes}.")
    if args is None:
        obj = cls()
    else:
        obj = cls(**args)
    return obj


def get_model(config) -> torch.nn.Module:
    prefixes = ["_creator.models.", "torchvision.models."]
    model = locate_and_init(config["model"]["class_name"], prefixes, config["model"].get("args", dict()))
    if config["model"]["load_checkpoint"]:
        model = load_checkpoint(model, config["model"]["checkpoint_path"], config["model"]["param_key"], config["device"])
    model.to(config["device"])
    model.eval()
    return model


def get_transformations(config, model=None):
    if config["transformations"].get("from_model", False):
        tfms, norm = model.get_transforms()
    else:
        prefixes = ["_creator.transformations."]
        tfms = locate_and_init(config["transformations"]["class_name"], prefixes, config["transformations"]["args"])
        tfms = tfms.tfms
        norm = None
        if config["transformations"]["norm"]["apply_norm"]:
            norm = tf.Normalize(
                mean=config["transformations"]["norm"]["mean"], std=config["transformations"]["norm"]["std"]
            )
    return tfms, norm


def get_dataset(config, tfms, norm=None):
    prefixes = ["_creator.datasets.", "torchvision.datasets."]
    dset_dict = config["datasets"][config["vis_type"]]
    args_dict = dset_dict["args"]
    args_dict["class_names"] = config["class_names"]
    if len(tfms) == 0:
        tfms = (lambda x: x,)
    if norm is not None:
        args_dict["transform"] = tf.Compose([*tfms, norm])
    else:
        args_dict["transform"] = tf.Compose([*tfms])
    args_dict["cat_col_names"] = config.get("extra_data_columns", dict()).get("categorical", [])
    dataset = locate_and_init(dset_dict["class_name"], prefixes, args_dict)
    return dataset


def setup_inner_dir(save_dir, config_path, continue_from):
    make_new = continue_from is None
    if make_new:
        timestamp = "{:%Y-%m-%d__%H-%M-%S}".format(datetime.datetime.now())
    else:
        timestamp = continue_from
    inner_dir = save_dir/timestamp
    inner_dir.mkdir(exist_ok=not make_new)
    try:
        shutil.copy(str(config_path), str(inner_dir))
    except shutil.SameFileError:
        pass
    return inner_dir


def get_single_pos_slice(full_act_shape, entry, i):
    if entry == "all":
        end_idx = full_act_shape[i]
        return slice(0, end_idx)
    elif "-" in entry:
        # range stuff
        start_idx, end_idx = map(int, entry.split("-"))
        return slice(start_idx, end_idx)
    elif hasattr(entry, "__getitem__"):
        # list of individual numbers
        return tuple(sorted(entry))
    else:
        raise ValueError(f"Invalid positional info {entry}.")


def get_pos_slices(full_act_shape, pos_info):
    """
    full_act_shape: activation shape without the batch dimension
    pos_info: positional information from the config file
    """
    needs_meshgrid = [False for _ in range(len(pos_info))]
    for i, entry in enumerate(pos_info):
        if hasattr(entry, "__getitem__") and not isinstance(entry, str): # list/tuple of some sort
            # do the entries form a range? (If yes, they can be represented by a slice instead of a meshgrid)
            entry = sorted(entry)
            entry_range = list(range(entry[0], entry[-1]+1))
            forms_range = True
            for range_elem in entry_range:
                if range_elem not in entry:
                    forms_range = False
                    break
            needs_meshgrid[i] = not forms_range
    # if a meshgrid is required, from which to which dimension does it have to span?
    if True in needs_meshgrid:
        meshgrid_start = needs_meshgrid.index(True)
        meshgrid_end = len(needs_meshgrid) - list(reversed(needs_meshgrid)).index(True) # end is exclusive
    else:
        meshgrid_start = len(pos_info)
        meshgrid_end = len(pos_info)

    # based on this info, actually build up the slices/meshgrid
    pos_slices = []
    for i, entry in enumerate(pos_info[:meshgrid_start]):
        pos_slices += [get_single_pos_slice(full_act_shape, entry, i)]
    if meshgrid_start < len(pos_info):
        # turn all entries between meshgrid_start and meshgrid_end into lists of indices
        pos_info_lst = list(pos_info) # pos_info is immutable which isn't great here
        for i, entry in list(enumerate(pos_info))[meshgrid_start:meshgrid_end]:
            if entry == "all":
                end_idx = full_act_shape[i]
                pos_info_lst[i] = list(range(0, end_idx))
            elif "-" in entry:
                start_idx, end_idx = map(int, entry.split("-"))
                pos_info_lst[i] = list(range(start_idx, end_idx))
        # create and add the meshgrid
        pos_slices += torch.meshgrid(
            list(map(
                torch.tensor,
                pos_info_lst[meshgrid_start:meshgrid_end]
            )),
            indexing="ij")
    for i, entry in list(enumerate(pos_info))[meshgrid_end:]: # only use elements in the back, but keep the original indices
        pos_slices += [get_single_pos_slice(full_act_shape, entry, i)]
    return pos_slices


def get_pos_string(pos_info):
    """
    Stringify the info in pos_info to use for folder/file/display names
    """
    pos_string = ""
    for entry in pos_info:
        if entry == "all":
            pos_string += "(all)_"
        elif "-" in entry:
            pos_string += f"({entry})_"
        elif hasattr(entry, "__getitem__"):
            pos_string += f"{tuple(sorted(entry))}_"
    pos_string = f"[{pos_string[:-1]}]"
    return pos_string


class Welford:
    # see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self):
        self.count = 0
        self.mean = None
        self.M2 = None

    def update(self, new_value): # new_value should start with a batch dimension and also probably be a torch.tensor
        count_old = self.count
        count_new = len(new_value)
        mean_new = new_value.mean(dim=0)
        if self.mean is None:
            self.mean = mean_new
            self.count = count_new
            self.M2 = ((new_value - self.mean[None, :])**2).sum(dim=0)
        else:
            M2_new = ((new_value - self.mean[None, :])**2).sum(dim=0)
            delta = mean_new - self.mean
            self.count = count_old + count_new
            self.mean = self.mean + delta*(count_new/self.count)
            self.M2 = self.M2 + M2_new + delta**2 * ((count_new*count_old)/self.count)

    def get_results(self):
        if self.count < 2:
            mean = None
            variance = None
            sample_variance = None
        else:
            mean = self.mean
            variance = self.M2/self.count
            sample_variance = self.M2/(self.count-1)
        return {"mean": mean, "variance": variance, "sample_variance": sample_variance}


def merge_dim_output(dim0, dim1, use_all):
    if dim0 is None:
        dim0 = []
    if dim1 is None:
        dim1 = []
    if isinstance(dim0, np.integer):
        dim0 = int(dim0)
    if isinstance(dim1, np.integer):
        dim1 = int(dim1)

    if use_all:
        if dim0 == "all" or dim1 == "all":
            return "all"
    else:
        if dim0 == "all":
            dim0 = []
        if dim1 == "all":
            dim1 = []

    if isinstance(dim0, list) and isinstance(dim1, list):
        return sorted(list(set(dim0 + dim1)))
    elif isinstance(dim0, list):
        return sorted(list(set(dim0 + [dim1])))
    elif isinstance(dim1, list):
        return sorted(list(set([dim0] + dim1)))
    else:
        return sorted(list({dim0, dim1}))

merge_dim_output_ufunc = np.frompyfunc(functools.partial(merge_dim_output, use_all=False), nin=2, nout=1)
merge_dim_output_use_all_ufunc = np.frompyfunc(functools.partial(merge_dim_output, use_all=True), nin=2, nout=1)


def get_random_batches(num_elems, batch_size, device):
    """
    Shuffles the indices [0, 1, 2, ... num_elems-1] around and chops the result into batch_size-sized tensors
    (last element is smaller if num_elems is not divisible by batch_size)
    """
    indices = torch.randperm(num_elems, dtype=torch.long, device=device)
    num_batches = num_elems // batch_size
    num_batches += 1 if num_elems % batch_size != 0 else 0
    indices_chopped = []
    for b in range(num_batches):
        batch_start = b*batch_size
        batch_end = batch_start + batch_size
        if batch_end > num_elems: batch_end = num_elems
        indices_chopped += [indices[batch_start:batch_end]]
    return indices_chopped


def map_to_range(val, old_min, old_max, new_min, new_max):
    return ((val - old_min) * ((new_max - new_min) / (old_max - old_min))) + new_min


# from: https://github.com/greentfrapp/lucent/issues/45#issuecomment-2106395997
# lucent apparently doesn't clean up the forward_hooks it registers, slowing down
# all subsequent uses of render_vis --> use this function to clean them up.
def remove_all_forward_hooks(model):
    for _, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks = OrderedDict()
            remove_all_forward_hooks(child)