import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from _creator import utils


def render_img_grid(cell_imgs_folder, save_path, max_size, num_cells_y, num_cells_x=None):
    """
    Saves an image grid of the images in cell_imgs_folder to save_path, scaled down to a maximum size of
    max_size x max_size pixels.

    Expects the cell images to be named like <y_pos>_<x_pos>.png
    """
    if num_cells_x is None:
        num_cells_x = num_cells_y
    num_cells = np.array([num_cells_y, num_cells_x])
    cell_imgs_paths = [p for p in cell_imgs_folder.iterdir() if p.suffix == ".png"]
    cell_img = cv2.imread(str(cell_imgs_paths[0]))
    unscaled_cell_size = np.array(cell_img.shape[:2])
    cell_ch = cell_img.shape[2]
    unscaled_grid_size = unscaled_cell_size * num_cells
    unscaled_big_dim = max(unscaled_grid_size)
    if unscaled_big_dim > max_size:
        img_sf = max_size / unscaled_big_dim
    else:
        img_sf = 1
    scaled_grid_size = np.ceil(unscaled_grid_size * img_sf).astype(int)
    scaled_cell_size = (unscaled_cell_size * img_sf).astype(int)

    grid_img = np.zeros((*scaled_grid_size, cell_ch), dtype=cell_img.dtype) + 255
    for i, cell_img_path in enumerate(cell_imgs_paths):
        grid_pos = np.array(cell_img_path.stem.split("_"), dtype=int)
        offset = grid_pos * scaled_cell_size
        cell_img = cv2.imread(str(cell_img_path))
        if img_sf != 1:
            cell_img = cv2.resize(
                cell_img, (scaled_cell_size[1], scaled_cell_size[0]), interpolation=cv2.INTER_AREA
            )
        grid_img[offset[0] : offset[0] + scaled_cell_size[0], offset[1] : offset[1] + scaled_cell_size[1]] = (
            cell_img
        )
    cv2.imwrite(str(save_path), grid_img)


def render_attribution_grid(
    attributions,
    save_path,
    num_cells,
    grid_size,
    class_names,
    valid_idxs=None,
    attrib_scale=10000,
    class_colors=plt.cm.tab10.colors,
    txt=True
):
    cell_size = grid_size // num_cells
    blank_cell_img = np.zeros((cell_size, cell_size, 3), dtype=np.uint8) + 255
    grid_img = np.zeros((grid_size, grid_size, 3), dtype=np.uint8) + 255

    if valid_idxs is None:
        # attributions has shape [num_classes, num_cells, num_cells]
        valid_idxs = torch.nonzero(
            torch.ones((num_cells, num_cells)) > 0, as_tuple=False
        )  # [num_cells*num_cells, 2]
        attributions = attributions.reshape(
            (-1, num_cells * num_cells)
        )  # now [num_classes, num_cells*num_cells]
    else:
        # attributions has shape [num_valid_cells, num_classes]
        attributions = torch.movedim(attributions, (0,), (1,))  # now [num_classes, num_valid_cells]
    num_classes = attributions.shape[0]
    class_colors = [
        np.array((int(c[2] * 255), int(c[1] * 255), int(c[0] * 255)), dtype=grid_img.dtype)
        for c in class_colors
    ]
    class_overlays = np.zeros((num_classes, cell_size, cell_size, 3), dtype=grid_img.dtype)
    for i in range(num_classes):
        class_overlays[i] = class_colors[i % 10][None, None, :]
    cls_idxs = torch.argmax(attributions, dim=0)  # shape [num_valid_cells]
    attrib_softmaxed = torch.nn.functional.softmax(attrib_scale * attributions, dim=0)

    for i, idx in enumerate(valid_idxs):  # i indexes into attributions, y_idx and x_idx into the grid
        y_idx = idx[0].item()
        x_idx = idx[1].item()
        offset_y = y_idx * cell_size  # indexes into grid_img
        offset_x = x_idx * cell_size
        alpha = utils.map_to_range(
            attrib_softmaxed[cls_idxs[i], i], old_min=1 / num_classes, old_max=1.0, new_min=0.5, new_max=1.0
        ).item()
        # alpha = attrib_softmaxed[cls_idxs[i], i].item()
        cell_img = (1.0 - alpha) * blank_cell_img + alpha * class_overlays[cls_idxs[i]]
        if txt:
            cell_img = cv2.putText(
                cell_img,
                class_names[cls_idxs[i]],  # text to draw
                (0, cell_size - 1),  # bottom-left corner of the text string in the image
                cv2.FONT_HERSHEY_SIMPLEX,
                cell_size / 100,  # font scale,
                (255, 255, 255),  # text color
                1,  # font thickness
            )
        grid_img[offset_y : offset_y + cell_size, offset_x : offset_x + cell_size] = cell_img
    cv2.imwrite(str(save_path), grid_img)


def render_layout(layout, save_path, class_names, label_col, class_colors=plt.cm.tab10.colors):
    figsize = max(plt.rcParams["figure.figsize"])
    fig = plt.figure(figsize=(figsize, figsize))
    ax = fig.add_axes((0,0,1,1))
    for i, name in enumerate(class_names):
        cls_indices = label_col[
            label_col == name
        ].index.to_numpy()  # entry indices originally from this class
        ax.scatter(
            x=layout[cls_indices, 1],
            y=1 - layout[cls_indices, 0],
            s=1,
            marker=".",
            color=class_colors[i % 10],
            alpha=0.3
        )
    ax.set_xlim(0, layout[:, 1].max())
    ax.set_ylim(0, layout[:, 0].max())
    plt.axis("off")
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)


def create_cell_thumbnails(cell_imgs_folder, downscale_by=(4, 16)):
    cell_imgs_paths = [p for p in cell_imgs_folder.iterdir() if p.suffix == ".png"]
    downscale_folders = [cell_imgs_folder / f"div_{factor}" for factor in downscale_by]
    for p in downscale_folders:
        p.mkdir()
    for p in cell_imgs_paths:
        cell_img = cv2.imread(str(p))
        img_size = np.array(cell_img.shape[:2], dtype=int)
        scaled_img_sizes = [(img_size / factor).astype(int) for factor in downscale_by]
        for factor, scaled_img_size, downscale_folder in zip(
            downscale_by, scaled_img_sizes, downscale_folders
        ):
            scaled_img = cv2.resize(
                cell_img, (scaled_img_size[1], scaled_img_size[0]), interpolation=cv2.INTER_AREA
            )
            cv2.imwrite(str(downscale_folder / p.name), scaled_img)
