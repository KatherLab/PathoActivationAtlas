import torch
import umap
import sklearn
import numpy as np
from tqdm import tqdm
from lucent.optvis import render, param, transform, objectives
import matplotlib.pyplot as plt
import shutil

from _creator import utils, thumbnails, captum_fragments, objective_funcs


def load_layout_and_whiten_transform(act_dir, layout_from, save_dict_id):
    save_dict_path = act_dir/layout_from/f"grid_{save_dict_id}.pt"
    if save_dict_path.exists():
        save_dict = torch.load(save_dict_path)
        layout = save_dict["layout"]
        whiten_transform = save_dict["whiten_transform"]
        tqdm.write(f"Loaded layout from {save_dict_path}")
        return layout, whiten_transform
    else:
        tqdm.write(f"{save_dict_path} does not exist, creating new layout.")
        return None, None

def embed(activations, method, **method_kwargs):
    """
    Map activations into 2D space via UMAP or t-SNE
    """
    num_samples = len(activations)
    activations_np = np.array(activations.reshape((num_samples, -1)))
    min_percentile = method_kwargs.pop("min_percentile", 1)
    max_percentile = method_kwargs.pop("max_percentile", 99)
    relative_margin = method_kwargs.pop("relative_margin", 0.1)

    if method == "umap":
        default_args = {
            "n_components": 2,
            "verbose": True,
            "n_neighbors": 20,
            "min_dist": 0.01,
            "metric": "cosine"
        }
        method_kwargs = {**default_args, **method_kwargs}
        layout = umap.UMAP(**method_kwargs).fit_transform(activations_np)
    elif method == "tsne":
        # arguments from master thesis, except for the metric which is unspecified i think?
        default_args = {
            "n_components": 2,
            "verbose": True,
            "perplexity": 50.0,
            "learning_rate": 10,
            "metric": "cosine"
        }
        method_kwargs = {**default_args, **method_kwargs}
        layout = sklearn.manifold.TSNE(**method_kwargs).fit_transform(activations_np)
    else:
        raise ValueError(f"Unsupported embedding method {method}. Supported: 'umap', 'tsne'")

    # remove outliers by clipping, scale layout to between 0 and 1
    mins = np.percentile(layout, min_percentile, axis=0)
    maxs = np.percentile(layout, max_percentile, axis=0)
    mins -= relative_margin * (maxs - mins)
    maxs += relative_margin * (maxs - mins)
    layout = np.clip(layout, mins, maxs)
    layout -= layout.min(axis=0)
    layout /= layout.max(axis=0) # layout shape: [num_samples, 2]
    return torch.tensor(layout)


def avg_grid(layout, activations, attributions_default, num_cells, config, dataset):
    act_shape = activations.shape[1:]
    num_classes = attributions_default.shape[1]
    # num_cells-dimensions moved compared to old implementation
    avg_activations = torch.zeros((num_cells, num_cells, *act_shape)) + np.inf
    avg_attributions_default = torch.zeros((num_cells, num_cells, num_classes)) + np.inf
    cell_to_ids = dict() # for each cell, tensor of indices indicating inputs that belong to this cell
    # collect cell-wise input data for the columns in extra_data_column
    extra_data = {
        **{cat_col: {
            "classes": sorted(np.unique(dataset.df[cat_col].values).tolist()),
            "counts": torch.zeros((num_cells, num_cells, len(np.unique(dataset.df[cat_col].values))), dtype=torch.int)
        } for cat_col in config["extra_data_columns"]["categorical"]},
        **{cont_col: {
            "mean": torch.zeros((num_cells, num_cells)) + np.inf,
            "stddev": torch.zeros((num_cells, num_cells)) + np.inf,
            "median": torch.zeros((num_cells, num_cells)) + np.inf,
            "min": torch.zeros((num_cells, num_cells)) + np.inf,
            "max": torch.zeros((num_cells, num_cells)) + np.inf
        } for cont_col in config["extra_data_columns"]["continuous"]}
    }
    extra_columns = {
        col_name: {
            "data": dataset.df[col_name].values,
            "is_categorical": "counts" in extra_data[col_name]
        } for col_name in extra_data
    }
    counts = torch.zeros((num_cells, num_cells))
    step_size = 1/num_cells

    start_idx = [i*step_size for i in range(num_cells)]
    end_idx = [(i+1)*step_size for i in range(num_cells)]
    end_idx[-1] += 0.1 # epsilon to ensure elements at more or less exactly 1.0 aren't missed

    for y in range(num_cells):
        for x in range(num_cells):
            start_pt = torch.tensor([start_idx[y], start_idx[x]])
            end_pt = torch.tensor([end_idx[y], end_idx[x]])
            # find all embedded points that lie within y = [start_y, start_y) and x = [start_x, end_x)
            in_cell = (layout >= start_pt) * (layout < end_pt)
            in_cell = torch.all(in_cell, dim=1, keepdim=False) # should be of shape [num_samples] now, with entry True if within this cell
            counts[y,x] = torch.sum(in_cell)
            if counts[y,x] == 0:
                continue
            # calculate averages or gather data for this specific cell
            avg_activations[y,x,:] = torch.mean(activations[in_cell], dim=0)
            avg_attributions_default[y,x,:] = torch.mean(attributions_default[in_cell], dim=0)
            cell_to_ids[y,x] = torch.nonzero(in_cell).reshape((-1,))
            for col_name in extra_columns:
                cell_data = np.take(extra_columns[col_name]["data"], cell_to_ids[y,x], axis=0)
                if extra_columns[col_name]["is_categorical"]:
                    labels, label_counts = np.unique(cell_data, return_counts=True)
                    # if labels[i] == extra_data[col_name]["classes"][j], then label_idx[i] == j
                    label_idx = torch.zeros((len(labels),), dtype=torch.long)
                    for i, label in enumerate(labels):
                        label_idx[i] = extra_data[col_name]["classes"].index(label)
                    extra_data[col_name]["counts"][y,x,label_idx] = torch.tensor(label_counts, dtype=torch.int)
                else: # is continuous
                    extra_data[col_name]["mean"][y,x] = np.mean(cell_data)
                    extra_data[col_name]["stddev"][y,x] = np.std(cell_data)
                    extra_data[col_name]["median"][y,x] = np.median(cell_data)
                    extra_data[col_name]["min"][y,x] = np.min(cell_data)
                    extra_data[col_name]["max"][y,x] = np.max(cell_data)

    return avg_activations, avg_attributions_default, counts, cell_to_ids, extra_data


def whitening_transform(activations):
    """
    Provides the matrix used for whitening a layer's spacial activations
    See also: https://en.wikipedia.org/wiki/Whitening_transformation

    Adapted from:
    https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/activation-atlas/activation-atlas-simple.ipynb

    :param activations:         All spacial activations for a certain layer. torch.tensor of shape
                                [num_samples, num_channels]
    :return:                    The matrix used for whitening spacial activations
    """
    correl = torch.matmul(activations.t(), activations) / len(activations)
    try:
        transf = torch.inverse(correl)
    except RuntimeError:
        tqdm.write("[activation_atlas.py :: whitening_transform]: Correlation matrix is not invertible, using pseudoinverse instead.")
        transf = torch.pinverse(correl)
    return transf


def extract_activations_at_pos(model, layer_name, pos_slices):
    """
    model:          model wrapped by lucent
    layer_name      layer name, with dots already replaced with underscores
    pos_info:       List of slices to extract activations. None if convolutional layer.

    returns: extracted activations in the middle (for conv layers) or at the positions indicated by pos_info
    """
    layer = model(layer_name)
    bs = layer.shape[0]
    if pos_slices is None:  # conv layer with shape [bs, ch, y, x]
        current_activations = objectives._extract_act_pos(layer, x=None, y=None).reshape((bs, -1))  # [bs, ch]
    else:
        current_activations = layer[:, *pos_slices]
    return current_activations


def invert_activations(layer_name, model, target_activations, invert_size, transforms, num_steps,
                       batch_size, device, objective, pos_slices, optimizer, **objective_kwargs):
    """
    layer_name:         name of layer to invert activations for, dots should already be replaced with underscores
    model:              pytorch model
    target_activations: target activations of shape [num_samples, *(act_shape)] where act_shape indicates the shape
                        of the extracted features.
    invert_size:        output image size
    transforms:         transformations to apply to the target images before they go into the model
    num_steps:          Number of optimization steps for each output image
    batch_size:         Number of output images to optimize in any given step
    device:             string indicating which device to run the optimization on
    objective:          string indicating what objective to use during optimization. currently supported: direction_neuron_cossim
    pos_slices:         List of slices to extract activations. None if convolutional layer.
    optimizer:          One of:
                            - function that maps (params to optimize) to (optimizer object for these params)
                            - None (will use lucent's default optimizer)
    objective_kwargs:   Additional arguments for the objective
    """
    target_activations = target_activations.to(device)
    num_samples = target_activations.shape[0]
    num_batches = num_samples // batch_size
    num_batches += 1 if num_samples % batch_size != 0 else 0

    # parameterization of each cell image
    params_list = [None for i in range(num_batches)]
    image_f_list = [None for i in range(num_batches)]
    for b in range(num_batches):
        batch_start = b*batch_size
        batch_end = batch_start + batch_size
        if batch_end > num_samples:
            batch_end = num_samples
        bs = batch_end - batch_start
        image_f_list[b] = captum_fragments.NaturalImage(
            size=(int(invert_size[0]), int(invert_size[1])),
            batch=bs,
            parameterization=captum_fragments.FFTImage,
            decorrelate_init=True
        ).to(device)
        params_list[b] = image_f_list[b].parameters()

    # get the images that will be optimized in a given batch. each call to f() (so each call to image_f_batched) should
    # increment sample.batch and save the corresponding indices to sample.inds
    def sample(image_f_list, batch_size):
        sample.batch = 0
        # at the start of render_vis, lucent calls f() once without actually performing any optimization steps.
        # I'm using this flag to not increment the batch counter in that case
        sample.first_call_over = False
        def f():
            imgs = image_f_list[sample.batch]()
            batch_start = sample.batch * batch_size
            batch_end = batch_start + batch_size
            if batch_end > num_samples: batch_end = num_samples
            inds = torch.arange(batch_start, batch_end, dtype=torch.long)
            if sample.first_call_over:
                sample.inds = inds
                sample.batch += 1
                sample.batch = sample.batch % num_batches
            sample.first_call_over = True
            return imgs
        return f
    image_f_batched = sample(image_f_list, batch_size=batch_size)

    # params_list is a list of lists, concatenate into a 1-dimensional list to pass to lucent
    def param_f_batched():
        params = []
        for p in params_list:
            params += p
        return params, image_f_batched

    def get_activations(model, get_imgs, flatten=True):
        target_activation_batch = target_activations[sample.inds].to(device)
        bs = target_activation_batch.shape[0]
        current_activations = extract_activations_at_pos(model, layer_name, pos_slices)
        if flatten:
            current_activations = current_activations.reshape((bs, -1))
            target_activation_batch = target_activation_batch.reshape((bs, -1))
        batch_idx = (sample.batch - 1) % num_batches
        current_imgs = image_f_list[batch_idx]().to(device) if get_imgs else None
        return current_activations, target_activation_batch, current_imgs

    objective_func = getattr(objective_funcs, objective)
    def obj_func(model):
        return objective_func(model, get_activations, **objective_kwargs)
    obj = objectives.Objective(obj_func)

    results = render.render_vis(
        model=model,
        optimizer=optimizer,
        objective_f=obj,
        param_f=param_f_batched,
        transforms=transforms,
        thresholds=(num_steps*num_batches,),
        show_image=False,
        preprocess=False,
        progress=True,
        fixed_image_size=(int(invert_size[0]), int(invert_size[1]))
    )
    utils.remove_all_forward_hooks(model)
    imgs = torch.cat([image_f_list[b]().detach().cpu() for b in range(num_batches)], dim=0)
    imgs = imgs.permute(0,2,3,1).numpy()
    return imgs


def make_layer_atlas(layer_name, model, dataset, tfms, norm, config, atlas_dir, pos_info=None):
    """
    layer_name:         name of layer to create data for
    model:              pytorch model
    dataset:            dataset used in procuring the activations
    tfms, norm:         see create.py: tfms, norm = utils.get_transformations(config)
    config:             see create.py: save_dir, config = utils.open_config(args.config, args.vis_type)
    atlas_dir:          subfolder to save this atlas creation run to
    pos_info:           additional data indicating where in the layer the activations came from. None if convolutional layer.
    """
    pos_string = "" if pos_info is None else f" {utils.get_pos_string(pos_info)}"
    layer_dir = atlas_dir / f"{layer_name}{pos_string}"
    if (atlas_dir/f"grid_{layer_name}{pos_string}.pt").exists():
        tqdm.write(f"-------------- Skipping layer {layer_name}{pos_string} --------------")
        return
    tqdm.write(f"-------------- Processing layer {layer_name}{pos_string} --------------")

    # clean up any partial writes for this layer
    if layer_dir.exists():
        shutil.rmtree(str(layer_dir))

    activation_dict = torch.load(atlas_dir.parent/f"{layer_name}{pos_string}.pt") # keys: "layer_name", "act", "attr_default", "input_size", "full_act_size", "pos_slices"
    layer_name_lucent = layer_name.replace(".", "_")

    # embedding - check if we should load a preexisting layout first
    whiten_transform = None
    embedding_layout = None
    if config["layout_from"] is not None:
        embedding_layout, whiten_transform = load_layout_and_whiten_transform(
            atlas_dir.parent, config["layout_from"], f"{layer_name}{pos_string}"
        )
    if embedding_layout is None:
        embedding_kwargs = config["atlas_args"]["embedding_args"]
        if embedding_kwargs is None:
            embedding_kwargs = dict()
        embedding_layout = embed(activation_dict["act"], config["atlas_args"]["embedding"], **embedding_kwargs)

    # average activations and attributions across grid cells.
    # also record which idx went into which cell and the effect on the data in extra_data_columns
    num_cells = config["atlas_args"]["num_cells"]
    tqdm.write("Starting avg_grid...")
    avg_activations, avg_attributions_default, counts, cell_to_ids, extra_data = avg_grid(
        embedding_layout, activation_dict["act"], activation_dict["attr_default"],
        num_cells, config, dataset
    )

    # get and apply the whitening transformation
    if pos_info is None and config["atlas_args"]["whiten_conv"]:
        if whiten_transform is None:
            tqdm.write("Creating whitening transform...")
            whiten_transform = whitening_transform(activation_dict["act"])
        tqdm.write("Applying whitening transform...")
        avg_activations = torch.matmul(avg_activations, whiten_transform)
    elif pos_info is not None and config["atlas_args"]["whiten_other"]:
        if whiten_transform is None:
            tqdm.write("Creating whitening transform...")
            act_shape = avg_activations.shape[2:]
            num_samples = activation_dict["act"].shape[0]
            whiten_transform = whitening_transform(activation_dict["act"].reshape((num_samples, -1)))
        # reshape from [num_cells, num_cells, *act_shape] to [num_cells*num_cells, -1], whiten, then undo reshaping
        tqdm.write("Applying whitening transform...")
        avg_activations = torch.matmul(
            avg_activations.reshape((num_cells*num_cells, -1)),
            whiten_transform
        ).reshape((num_cells, num_cells, *act_shape))
    else:
        whiten_transform = None

    # filter out activations with count == 0, flatten grid dimension
    valid_activations = avg_activations[counts > 0] # [num_valid_acts, *act_shape]
    valid_attributions_default = avg_attributions_default[counts > 0]
    valid_idxs = torch.nonzero(counts > 0, as_tuple=False) # [num_valid_acts, 2]

    norm = [] if norm is None else [norm]
    transforms = tfms + [
        # some transformations to make the optimized image transformation robust
        # These are the same transforms as in the lucid notebook
        transform.pad(2, mode="constant", constant_value=1.0),
        transform.jitter(4),
        transform.jitter(4),
        transform.jitter(8),
        transform.jitter(8),
        transform.jitter(8),
        transform.random_scale([0.995 ** n for n in range(-5, 80)] + [0.998 ** n for n in 2 * list(range(20, 40))]),
        transform.random_rotate(list(range(-20, 20)) + list(range(-10, 10)) + list(range(-5, 5)) + 5 * [0]),
        transform.jitter(2)
    ] + norm

    opt_objective_args = config["atlas_args"]["opt_objective_args"]
    if opt_objective_args is None:
        opt_objective_args = dict()

    optimizer_init = None
    optimizer_config = config["atlas_args"].get("optimizer", None)
    if optimizer_config is not None:
        optimizer_init = lambda params: utils.locate_and_init(
            cls_name=optimizer_config["class_name"],
            prefixes=["torch.optim.", "_creator.optimizers."],
            args={"params": params, **optimizer_config["args"]}
        )

    imgs = invert_activations(
        layer_name=layer_name_lucent,
        model=model,
        target_activations=valid_activations,
        invert_size=activation_dict["input_size"],
        transforms=transforms,
        num_steps=config["atlas_args"]["num_steps"],
        batch_size=config["atlas_args"]["batch_size"],
        device=config["device"],
        objective=config["atlas_args"]["opt_objective"],
        pos_slices=activation_dict["pos_slices"],
        optimizer=optimizer_init,
        **opt_objective_args
    )

    layer_dir.mkdir(exist_ok=False)
    for i, idx in tqdm(enumerate(valid_idxs), total=len(valid_idxs), desc="Saving images to disk..."):
        y = idx[0].item()
        x = idx[1].item()
        fname = f"{y}_{x}.png"
        plt.imsave(layer_dir/f"{fname}", imgs[i])
    tqdm.write("Saving previews and thumbnails...")
    thumbnails.render_img_grid(
        layer_dir, layer_dir.parent/f"{layer_name}{pos_string}.png", num_cells_y=num_cells, max_size=(2**15)-1
    ) # from opencv documentation: "By default number of pixels must be less than 2^30."
    thumbnails.render_img_grid(
        layer_dir, layer_dir.parent/f"{layer_name}{pos_string}_small.png", num_cells_y=num_cells, max_size=2000
    )
    thumbnails.render_attribution_grid(
        valid_attributions_default, layer_dir.parent/f"{layer_name}{pos_string}_attributions_default.png", num_cells=num_cells,
        grid_size=2000, class_names=config["class_names"], valid_idxs=valid_idxs
    )
    thumbnails.render_layout(
        embedding_layout, layer_dir.parent/f"{layer_name}{pos_string}_layout.png",
        class_names=config["class_names"], label_col=dataset.label_col
    )
    thumbnails.create_cell_thumbnails(layer_dir, config["thumbnail_div_levels"])

    save_dict = {
        "layout": embedding_layout,
        "avg_activations": avg_activations, # already whitened if applicable
        "avg_attributions_default": avg_attributions_default,
        "whiten_transform": whiten_transform,
        "num_cells": num_cells,
        "counts": counts,
        "cell_to_ids": cell_to_ids,
        "extra_data": extra_data
    }
    torch.save(save_dict, atlas_dir/f"grid_{layer_name}{pos_string}.pt")


def create(model, dataset, tfms, norm, config, save_dir, config_path, **kwargs):
    atlas_dir = utils.setup_inner_dir(save_dir, config_path, config["continue_from"])
    for layer_name in config["layers"]:
        pos_infos = config["layers"][layer_name]
        for pos_info in pos_infos:
            _pos_info = None if pos_info == "conv" else pos_info
            with torch.no_grad():
                torch.cuda.empty_cache()
            make_layer_atlas(layer_name, model, dataset, tfms, norm, config, atlas_dir, _pos_info)
    shutil.copy(config_path, atlas_dir/"atlas")
    return atlas_dir