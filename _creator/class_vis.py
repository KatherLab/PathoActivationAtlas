import torch
import numpy as np
from tqdm import tqdm
from lucent.optvis import render, transform, objectives
import matplotlib.pyplot as plt
import shutil

from _creator import utils, thumbnails, captum_fragments, objective_funcs


def make_class_vis(model, tfms, norm, config, vis_dir, target_idx):
    device = config["device"]
    layer_name = config["class_vis_args"]["logit_layer_name"].replace(".", "_")
    target_name = config["class_names"][target_idx]
    class_dir = vis_dir / target_name

    # skip finished classes / cleanup for partial results
    if (vis_dir / f"grid_{target_name}.pt").exists():
        tqdm.write(f"-------------- Skipping class {target_name} --------------")
        return
    tqdm.write(f"-------------- Processing class {target_name} --------------")
    if class_dir.exists():
        shutil.rmtree(str(class_dir))

    num_cells = config["class_vis_args"]["num_cells"]
    num_steps = config["class_vis_args"]["num_steps"]
    num_total = num_cells*num_cells
    batch_size = config["class_vis_args"]["batch_size"]
    num_batches = num_total // batch_size
    num_batches += 1 if num_total % batch_size != 0 else 0
    invert_size = config["class_vis_args"]["invert_size"]

    params_list = [None for i in range(num_batches)]
    image_f_list = [None for i in range(num_batches)]
    for b in range(num_batches):
        batch_start = b*batch_size
        batch_end = batch_start + batch_size
        if batch_end > num_total:
            batch_end = num_total
        bs = batch_end - batch_start
        image_f_list[b] = captum_fragments.NaturalImage(
            size=invert_size,
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
            if batch_end > num_total: batch_end = num_total
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

    def get_activations(model, get_imgs):
        logits = model(layer_name)
        target = torch.zeros((logits.shape[0],), dtype=torch.long) + target_idx
        target = target.to(device)
        batch_idx = (sample.batch - 1) % num_batches
        current_imgs = image_f_list[batch_idx]().to(device) if get_imgs else None
        return logits, target, current_imgs

    objective = config["class_vis_args"]["opt_objective"]
    objective_kwargs = config["class_vis_args"]["opt_objective_args"]
    if objective_kwargs is None:
        objective_kwargs = dict()
    objective_func = getattr(objective_funcs, objective)
    def obj_func(model):
        return objective_func(model, get_activations, **objective_kwargs)
    obj = objectives.Objective(obj_func)

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

    optimizer_init = None
    optimizer_config = config["class_vis_args"].get("optimizer", None)
    if optimizer_config is not None:
        optimizer_init = lambda params: utils.locate_and_init(
            cls_name=optimizer_config["class_name"],
            prefixes=["torch.optim.", "_creator.optimizers."],
            args={"params": params, **optimizer_config["args"]}
        )

    results = render.render_vis(
        model=model,
        optimizer=optimizer_init,
        objective_f=obj,
        param_f=param_f_batched,
        transforms=transforms,
        thresholds=(num_steps*num_batches,),
        show_image=False,
        preprocess=False,
        progress=True,
        fixed_image_size=invert_size
    )
    utils.remove_all_forward_hooks(model)
    imgs = torch.cat([image_f_list[b]().detach().cpu() for b in range(num_batches)], dim=0)
    imgs = imgs.permute(0,2,3,1).numpy()

    class_dir.mkdir(exist_ok=False)
    for i in range(len(imgs)):
        x = i % num_cells
        y = i // num_cells
        fname = f"{y}_{x}.png"
        plt.imsave(class_dir/fname, imgs[i])

    tqdm.write("Saving previews and thumbnails...")
    thumbnails.render_img_grid(
        class_dir, class_dir.parent/f"{target_name}.png", num_cells_y=num_cells, max_size=(2**15)-1
    ) # from opencv documentation: "By default number of pixels must be less than 2^30."
    thumbnails.render_img_grid(
        class_dir, class_dir.parent/f"{target_name}_small.png", num_cells_y=num_cells, max_size=2000
    )
    thumbnails.create_cell_thumbnails(class_dir, config["thumbnail_div_levels"])
    save_dict = {
        "num_cells": num_cells
    }
    torch.save(save_dict, vis_dir / f"grid_{target_name}.pt")

def create(model, tfms, norm, config, save_dir, config_path, **kwargs):
    vis_dir = utils.setup_inner_dir(save_dir, config_path, config["continue_from"])
    for target_idx in range(len(config["class_names"])):
        with torch.no_grad():
            torch.cuda.empty_cache()
        make_class_vis(model, tfms, norm, config, vis_dir, target_idx)
    shutil.copy(config_path, vis_dir/"class_vis")
    return vis_dir