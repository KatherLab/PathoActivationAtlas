from pathlib import Path
import lucent.misc.channel_reducer
from tqdm import tqdm
import shutil
import torch
from lucent.optvis import render, param, transform, objectives
import matplotlib.pyplot as plt
import numpy as np

from _creator import utils, thumbnails, objective_funcs


def make_layer_actgrid(layer_name, model, dataset, img_idx, tfms, norm, config, img_dir):
    """
    layer_name:         name of layer to create data for
    model:              pytorch model
    dataset:            dataset used to produce the activations
    img_idx:            current image's index in the dataset's dataframe
                        --> img_row = dataset.df.iloc[img_idx] --> img_row.index for keys, img_row[key] for value
    tfms, norm:         see create.py: tfms, norm = utils.get_transformations(config)
    config:             see create.py: save_dir, config = utils.open_config(args.config, args.vis_type)
    img_dir:            subfolder to save the activation grid data for image i to
    """
    layer_dir = img_dir/layer_name
    img_name = img_dir.name
    if (img_dir/f"grid_{layer_name}.pt").exists():
        tqdm.write(f"-------------- Skipping {img_name}: {layer_name} --------------")
        return
    tqdm.write(f"-------------- Processing {img_name}: {layer_name} --------------")

    # clean up partial writes
    if layer_dir.exists():
        shutil.rmtree(str(layer_dir))

    # first parent: timestamped dir, second parent: actgrid dir
    activation_dict = torch.load(img_dir.parent.parent/f"{layer_name}.pt")
    activations = activation_dict["act"][img_idx]
    attributions_default = activation_dict["attr_default"][img_idx]
    layer_name_lucent = layer_name.replace(".", "_")
    actgrid_args = config["actgrid_args"]

    # determine invert_size from original input size and size of the activation
    tol_factor = actgrid_args.get("invert_size_tol_factor", 3)
    min_size = actgrid_args.get("invert_size_min", 32)
    input_size = activation_dict["input_size"] # shape [y, x]
    act_size = activation_dict["full_act_size"] # shape [ch, y, x]
    invert_size = (
        min(max(int(np.ceil(tol_factor*input_size[0] / act_size[1])), min_size), input_size[0]),
        min(max(int(np.ceil(tol_factor*input_size[1] / act_size[2])), min_size), input_size[1])
    )

    # collect data from the dataframe for later use in the viewer
    img_row = dataset.df.iloc[img_idx]
    extra_data = {
        "name": img_name,
        "label": dataset.label_col[img_idx],
        **{col_name: img_row[col_name] for col_name in config["extra_data_columns"]["categorical"]},
        **{col_name: img_row[col_name] for col_name in config["extra_data_columns"]["continuous"]}
    }

    # same transformations as atlas.py
    norm = [] if norm is None else [norm]
    transforms = tfms + [
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

    # reduce number of channels to num_groups via non-negative matric factorization
    y_grid, x_grid = activations.shape[1:3]
    target_activations = activations.permute(1,2,0) # [y, x, ch]
    target_activations = target_activations.view(-1, target_activations.shape[-1]) # [y*x, ch]
    acts_np = target_activations.cpu().numpy()
    # if there are negative values: subtract lowest values from all entries to ensure no negative values
    # (basically just shift the whole thing into a positive range)
    if np.any((acts_np < 0)):
        acts_np -= np.min(acts_np)
    num_cells = acts_np.shape[0]
    num_groups = config["actgrid_args"]["num_groups"]
    reducer = lucent.misc.channel_reducer.ChannelReducer(num_groups, "NMF", max_iter=actgrid_args.get("nmf_max_iter", 5000))
    groups = reducer.fit_transform(acts_np)
    groups /= groups.max(0) # ???
    groups = torch.from_numpy(groups).to(config["device"])

    # parameterization of the group images
    groups_params, groups_image_f = param.fft_image([num_groups, 3, *invert_size])
    # parameterization of each individual image (y*x many)
    cells_params, cells_image_f = param.fft_image([num_cells, 3, *invert_size])
    group_weight = actgrid_args["group_weight"] # cell weight will be 1.0 minus this

    # actual grid images from mixing the images of both parameterizations
    def image_f():
        groups_images = groups_image_f()
        cells_images = cells_image_f()
        group_sums = torch.zeros((num_cells, 3, *invert_size)).to(groups_images.device)
        for j in range(num_groups):
            group_sums += groups[:, j, None, None, None] * groups_images[j]
        imgs = (1.0 - group_weight)*cells_images + group_weight*group_sums
        return imgs
    image_f = param.to_valid_rgb(image_f, decorrelate=True)

    # sample the cells that will actually get optimized. In contrast to the old implementation, I won't
    # do this randomly but rather make sure all cells are evenly optimized, similar to how it works for
    # the activation atlas
    batch_size = actgrid_args["batch_size"]
    num_batches = num_cells // batch_size
    num_batches += 1 if num_cells % batch_size != 0 else 0
    def sample(image_f, batch_size):
        sample.batch = 0
        sample.first_call_over = False
        sample.batch_pool = utils.get_random_batches(num_cells, batch_size, "cpu")
        def f():
            imgs = image_f()
            inds = sample.batch_pool[sample.batch]
            if sample.first_call_over:
                sample.inds = inds
                sample.batch += 1
                # replenish sample pool here if sample.batch >= num_batches
                if sample.batch >= num_batches:
                    sample.batch_pool = utils.get_random_batches(num_cells, batch_size, "cpu")
                    sample.batch = sample.batch % num_batches
            else:
                sample.first_call_over = True
            inputs = imgs[inds]
            return inputs
        return f
    image_f_sampled = sample(image_f, batch_size=batch_size)

    def get_activations(model, get_imgs):
        pred = model(layer_name_lucent)  # [batch_size, num_channels, y, x]
        target = target_activations[sample.inds].to(pred.device)  # [batch_size, num_channels]
        target = target[:, :, None, None]  # [batch_size, num_channels, 1, 1]
        current_imgs = image_f()[sample.inds].to(pred.device) if get_imgs else None
        return pred, target, current_imgs

    # objective function
    objective_func = getattr(objective_funcs, actgrid_args.get("opt_objective", "dot"))
    objective_args = actgrid_args.get("opt_objective_args", dict())
    def obj_func(model):
        return objective_func(model, get_activations, **objective_args)
    obj = objectives.Objective(obj_func)

    optimizer_init = None
    optimizer_config = actgrid_args.get("optimizer", None)
    if optimizer_config is not None:
        optimizer_init = lambda params: utils.locate_and_init(
            cls_name=optimizer_config["class_name"],
            prefixes=["torch.optim.", "_creator.optimizers."],
            args={"params": params, **optimizer_config["args"]}
        )

    # concatenate params so they all get optimized
    def param_f():
        params = list(groups_params) + list(cells_params)
        return params, image_f_sampled

    # invert activations
    num_steps = actgrid_args["num_steps"]
    input_size = activation_dict["input_size"]
    results = render.render_vis(
        model=model,
        optimizer=optimizer_init,
        objective_f=obj,
        param_f=param_f,
        transforms=transforms,
        thresholds=(num_steps*num_batches,),
        show_image=False,
        preprocess=False,
        progress=True,
        fixed_image_size=(int(input_size[0]), int(input_size[1]))
    )

    # save images and data to disk
    imgs = image_f().cpu().detach().permute(0,2,3,1).numpy()
    layer_dir.mkdir()
    for i in range(num_cells):
        x = i % x_grid
        y = i // x_grid
        fname = f"{y}_{x}.png"
        plt.imsave(layer_dir/fname, imgs[i])
    tqdm.write("Saving previews and thumbnails...")
    thumbnails.render_img_grid(
        layer_dir, img_dir/f"{layer_name}.png", num_cells_y=int(np.sqrt(num_cells)), max_size=(2**15)-1
    ) # from opencv documentation: "By default number of pixels must be less than 2^30."
    thumbnails.render_img_grid(
        layer_dir, img_dir/f"{layer_name}_small.png", num_cells_y=int(np.sqrt(num_cells)), max_size=2000
    )
    thumbnails.render_attribution_grid(
        attributions_default, layer_dir.parent/f"{layer_name}_attributions_default.png", num_cells=int(np.sqrt(num_cells)),
        grid_size=2000, class_names=config["class_names"]
    )
    thumbnails.create_cell_thumbnails(layer_dir, config["thumbnail_div_levels"])

    save_dict = {
        "activations": activations,
        "attributions_default": attributions_default,
        "num_cells": int(np.sqrt(num_cells)),
        "img_idx": img_idx,
        "extra_data": extra_data
    }
    torch.save(save_dict, img_dir/f"grid_{layer_name}.pt")


def create(model, dataset, tfms, norm, config, save_dir, config_path, **kwargs):
    timestamped_dir = utils.setup_inner_dir(save_dir, config_path, config["continue_from"])
    # folder structure: actgrid --> timestamped folder --> img_id --> same stuff as for an individual atlas basically
    for i in range(len(dataset)):
        if dataset.img_root is not None:
            img_path = dataset.img_root/dataset.path_col[i]
        else:
            img_path = Path(dataset.path_col[i])
        img_name = img_path.name
        img_dir = timestamped_dir/img_name
        img_dir.mkdir(exist_ok=True)
        for layer_name in config["layers"]:
            pos_infos = config["layers"][layer_name]
            if "conv" in pos_infos:
                with torch.no_grad():
                    torch.cuda.empty_cache()
                make_layer_actgrid(layer_name, model, dataset, i, tfms, norm, config, img_dir)
        # copy over the input image
        shutil.copy(img_path, img_dir/f"input_image{img_path.suffix}")
        shutil.copy(config_path, img_dir/"actgrid")
    return timestamped_dir
