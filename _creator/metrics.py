from pathlib import Path
import torch
from torchvision import transforms as tf
from dreamsim import dreamsim
from tqdm import tqdm
import pandas as pd
from natsort import index_natsorted
import numpy as np
import shutil
from sklearn import covariance

from _creator import utils, record, datasets, custom_lpips


@torch.no_grad()
def nearest_neighbor_stats(dist_to_imgs, num_nn, orig_labels, num_classes):
    num_gen, num_orig = dist_to_imgs.shape
    num_nn = min(num_nn, num_orig)
    nn_dists, nn_indices = torch.topk(
        dist_to_imgs, k=num_nn, dim=1, largest=False, sorted=True
    )  # shape of each of these: (num_gen, num_nn)
    nn_labels = torch.vmap(lambda x: orig_labels[x])(nn_indices)  # shape (num_gen, num_nn)
    nn_label_counts = torch.zeros((num_gen, num_classes), dtype=torch.int)
    # loop of shame, but idk how to vectorize this
    for g in range(num_gen):
        labels, counts = torch.unique(nn_labels[g], return_counts=True)
        for l, c in zip(labels, counts):
            nn_label_counts[g, l] = c
    nn_label_counts_rel = nn_label_counts / num_nn
    return nn_label_counts, nn_label_counts_rel, nn_dists, nn_indices


@torch.no_grad()
def lpips_dist(
    orig_dloader,
    gen_dloader,
    model_id,
    vis_layer_id,
    lpips_model,
    timestamp_dir,
    device,
    orig_labels,
    num_classes,
):
    orig_dir = timestamp_dir / "activations4metrics" / model_id
    gen_dir = timestamp_dir / vis_layer_id / "activations4metrics" / model_id
    num_gen = len(gen_dloader.dataset)
    num_orig = len(orig_dloader.dataset)
    gen_batch_size = gen_dloader.batch_size
    orig_batch_size = orig_dloader.batch_size
    metric_layers = lpips_model.target_layers

    def get_acts(imgs, act_dir, batch_idx, batch_size):
        file_paths = {layer: act_dir / f"{layer}___{batch_size}_{batch_idx}.pt" for layer in metric_layers}
        all_exist = all([path.exists() for layer, path in file_paths.items()])
        if not all_exist:
            act_dir.mkdir(parents=True, exist_ok=True)
            imgs = imgs.to(device)
            acts = lpips_model.get_activations(imgs)
            for layer, path in file_paths.items():
                if not path.exists():
                    torch.save(acts[layer].detach().cpu(), path)
        else:
            acts = dict()
            for layer, path in file_paths.items():
                acts[layer] = torch.load(path).to(device)
        return acts

    dist_to_imgs = torch.empty((num_gen, num_orig))
    orig_start = 0
    for orig_b, (orig_imgs, _) in tqdm(
        enumerate(orig_dloader), total=len(orig_dloader), desc=f"lpips: {model_id}, {vis_layer_id}"
    ):
        orig_bs = len(orig_imgs)
        orig_end = min(orig_start + orig_bs, num_orig)
        orig_acts = get_acts(orig_imgs, orig_dir, orig_b, orig_batch_size)
        gen_start = 0
        for gen_b, (gen_imgs, _) in enumerate(gen_dloader):
            gen_bs = len(gen_imgs)
            gen_end = min(gen_start + gen_bs, num_gen)
            gen_acts = get_acts(gen_imgs, gen_dir, gen_b, gen_batch_size)
            dist_to_imgs[gen_start:gen_end, orig_start:orig_end] = (
                lpips_model.get_distances(gen_acts, orig_acts).detach().cpu()
            )
            gen_start += gen_bs
        orig_start += orig_bs
    dist_to_cls = get_dist_to_cls(dist_to_imgs, orig_labels, num_classes)
    return dist_to_cls, dist_to_imgs


# technically *squared* mahalanobis distance
@torch.no_grad()
def mahalanobis_dist(orig_acts, gen_acts, orig_labels, num_classes):
    # reshape features to 1D
    orig_acts = orig_acts.reshape((orig_acts.shape[0], -1))
    gen_acts = gen_acts.reshape((gen_acts.shape[0], -1))
    dist_to_cls = torch.empty((len(gen_acts), num_classes))
    for c in range(num_classes):
        is_class = orig_labels == c
        idx_where_class = is_class.nonzero(as_tuple=True)[0]
        filtered_orig_acts = orig_acts[idx_where_class]
        mean = torch.mean(filtered_orig_acts, dim=0, keepdim=True)  # shape (1, feature_size)
        diff = (gen_acts - mean).T.to(torch.float64)  # shape (feature_size, num_gen)
        filtered_orig_acts = filtered_orig_acts.numpy(force=True).astype(np.float64)
        # the whole point of what follows is calculating (diff.T * covar^(-1) * diff) in a batched way, with num_gen
        # as the batch dimension
        # Estimate covariance matrix via LedoitWolf shrinkage for a somewhat better-conditioned matrix
        covar_ledoit = torch.from_numpy(
            covariance.LedoitWolf(store_precision=False).fit(filtered_orig_acts).covariance_
        ).to(
            diff.dtype
        )  # shape (feature_size, feature_size)
        # from the docs: "This function computes X = A.pinverse() @ B in a faster and more numerically stable way than
        # performing the computations separately."
        # - pseudoinverse because the cov matrix is practically never actually invertible
        # - driver=gelsd because that's recommended for non-full-rank, ill-conditioned matrices
        right = torch.linalg.lstsq(covar_ledoit, diff, driver="gelsd").solution
        # calculate batched dot product
        dist_to_cls[:, c] = (
            (diff * right).sum(dim=0).to(dist_to_cls.dtype)
        )  # shape (feature_size, num_gen) to (num_gen,)
    has_invalid_vals = not torch.all(torch.isfinite(dist_to_cls) & (dist_to_cls > 0))
    return dist_to_cls, has_invalid_vals


@torch.no_grad()
def dreamsim_dist(orig_acts, gen_acts, orig_labels, num_classes):
    # einops-ish: i1..., 1j... --> ij...
    gen_acts = gen_acts[:, None]
    dist_to_imgs = torch.empty((len(gen_acts), len(orig_acts)))
    # split up the computation for memory reasons
    for i in range(len(gen_acts)):
        dist_to_imgs[i] = 1 - torch.nn.functional.cosine_similarity(gen_acts[i], orig_acts, dim=-1)
    dist_to_cls = get_dist_to_cls(dist_to_imgs, orig_labels, num_classes)
    return dist_to_cls, dist_to_imgs


def get_dist_to_cls(dist_to_imgs, orig_labels, num_classes):
    dist_to_cls = torch.empty((dist_to_imgs.shape[0], num_classes))
    for c in range(num_classes):
        # i want: filtered dist_to_imgs, i.e. shape [len(gen_acts), len(label == c)]
        is_class = orig_labels == c
        idx_where_class = is_class.nonzero(as_tuple=True)[0]
        filtered_dist_to_imgs = dist_to_imgs[:, idx_where_class]
        dist_to_cls[:, c] = filtered_dist_to_imgs.mean(dim=1)  # mean along orig images
    return dist_to_cls


@torch.no_grad()
def record_dreamsim_features(actdir, desc_txt, model, dloader, device, **kwargs):
    num_samples = len(dloader.dataset)
    batch_size = dloader.batch_size
    act_path = actdir / "dreamsim.pt"
    if not act_path.exists():
        embeddings = torch.empty((num_samples, model.embed_size), dtype=torch.float32)
        for batch_idx, (imgs, _) in tqdm(enumerate(dloader), total=len(dloader), desc=desc_txt):
            imgs = imgs.to(device)
            start = batch_idx * batch_size
            end = min(start + batch_size, num_samples)
            embeddings[start:end] = model.embed(imgs).detach().cpu()
        torch.save(embeddings, act_path)


@torch.no_grad()
def record_wrapped_features(
    actdir,
    desc_txt,
    model,
    dloader,
    device,
    metric_layers,
    extra_forward_args,
    extra_forward_kwargs,
    **kwargs,
):
    for metric_layer in metric_layers:
        act_path = actdir / f"{metric_layer}.pt"
        if not act_path.exists():
            wrapped_model = record.ActivationOnlyModel(model, metric_layer, ("all",))
            wrapped_model.eval()
            desc = f"{desc_txt} | {metric_layer}"
            for imgs, _ in tqdm(dloader, total=len(dloader), desc=desc):
                imgs = imgs.to(device)
                wrapped_model(imgs, *extra_forward_args, **extra_forward_kwargs)
            save_dict = wrapped_model.get_record()
            torch.save(save_dict["act"], act_path)
            wrapped_model.reset_record()
            wrapped_model.deregister_layer_hook()


@torch.no_grad()
def record_metrics_features(config, timestamp_dir, model, transform, metric_layers, orig_dloader, model_id):
    if model_id.startswith("vis_"):
        extra_forward_kwargs = config["model"].get("extra_forward_kwargs", None) or dict()
        extra_forward_args = config["model"].get("extra_forward_args", None) or []
    elif model_id != "dreamsim":
        extra_forward_kwargs = (
            config["metrics"]["models"][model_id].get("extra_forward_kwargs", None) or dict()
        )
        extra_forward_args = config["metrics"]["models"][model_id].get("extra_forward_args", None) or []

    # activations for generated images
    tqdm.write("Recording activations for generated images...\n<model_id> | <vis_layer> | <metric_layer>")
    for layer_file in timestamp_dir.glob("grid_*.pt"):
        vis_layer_id = layer_file.stem[5:]
        if model_id == "dreamsim":
            gen_actdir = timestamp_dir / vis_layer_id / "activations4metrics"
        else:
            gen_actdir = timestamp_dir / vis_layer_id / "activations4metrics" / model_id
        gen_actdir.mkdir(exist_ok=True, parents=True)
        gen_dset = datasets._ImageFolderDataset(
            img_folder=timestamp_dir / vis_layer_id, transform=tf.Compose(transform)
        )
        gen_dloader = torch.utils.data.DataLoader(
            gen_dset,
            batch_size=config["extraction_batch_sizes"]["metrics"],
            shuffle=False,
            num_workers=config["num_dataloader_workers"],
        )
        record_kwargs = {
            "actdir": gen_actdir,
            "desc_txt": f"{model_id} | {vis_layer_id}",
            "model": model,
            "dloader": gen_dloader,
            "num_samples": len(gen_dset),
            "metric_layers": metric_layers,
            "device": config["device"],
        }
        if model_id == "dreamsim":
            record_dreamsim_features(**record_kwargs)
        else:
            record_wrapped_features(
                extra_forward_args=extra_forward_args,
                extra_forward_kwargs=extra_forward_kwargs,
                **record_kwargs,
            )

    # activations for original images
    tqdm.write("Recording activations for real images...\n<model_id> | <metric_layer>")
    if model_id == "dreamsim":
        orig_actdir = timestamp_dir / "activations4metrics"
    else:
        orig_actdir = timestamp_dir / "activations4metrics" / model_id
    orig_actdir.mkdir(exist_ok=True, parents=True)
    record_kwargs = {
        "actdir": orig_actdir,
        "desc_txt": f"{model_id}",
        "model": model,
        "dloader": orig_dloader,
        "num_samples": len(orig_dloader.dataset),
        "metric_layers": metric_layers,
        "device": config["device"],
    }
    if model_id == "dreamsim":
        record_dreamsim_features(**record_kwargs)
    else:
        record_wrapped_features(
            extra_forward_args=extra_forward_args, extra_forward_kwargs=extra_forward_kwargs, **record_kwargs
        )


def calculate_metrics(config, timestamp_dir, vis_model, vis_tfms):
    # set up dict that allows me to use the init methods in utils.py for the objects under the metrics key
    init_config = {
        "datasets": {"metrics": config["metrics"]["dataset"]},
        "device": config["device"],
        "vis_type": "metrics",
        "class_names": config["class_names"],
    }
    metrics = []
    for metric in config["metrics"]["to_calculate"]:
        if metric.startswith(("mahalanobis", "lpips_nolin")):
            if metric.endswith("_vis"):
                metrics += [f"{metric.replace('_vis', '')} | vis_model"]
            else:
                metrics += [f"{metric} | {model_id}" for model_id in config["metrics"].get("models", [])]
        else:
            metrics += [metric]
    for metric_idx, metric in (
        metric_pbar := tqdm(enumerate(metrics), total=len(metrics), desc="Calculating metrics")
    ):
        metric_pbar.set_postfix_str(metric)
        # keep vis_model on the cpu unless needed to save some VRAM
        vis_model = vis_model.to("cpu")
        # allow previous model to be garbage collected
        model = None
        # initialize model (if applicable), transformations and the dataset/dataloader for the original data
        if metric.startswith(("mahalanobis", "lpips_nolin")):
            model_id = metric.split(sep=" | ")[-1]
            if model_id == "vis_model":
                model = vis_model.to(config["device"]).eval()
                tfms = vis_tfms
            else:
                init_config["model"] = config["metrics"]["models"][model_id]
                init_config["transformations"] = config["metrics"]["transformations"].get(model_id, None)
                model = utils.get_model(init_config)
                tfms, norm = utils.get_transformations(init_config, model)
                tfms = tfms + [tf.ToTensor(), norm]
        elif metric == "dreamsim":
            model, _ = dreamsim(
                pretrained=True,
                device=config["device"],
                cache_dir=str(Path(torch.hub.get_dir()) / "dreamsim"),
            )
            # vvv tfms grabbed from dreamsim method @ dreamsim/model.py
            tfms = [tf.Resize((224, 224), interpolation=tf.InterpolationMode.BICUBIC), tf.ToTensor()]
            model_id = "dreamsim"
        else:  # metric == lpips_{alex, vgg, squeeze}
            # lpips expects images in range [-1, 1]
            tfms = [
                tf.Resize(config["metrics"]["lpips_size"]),
                tf.ToTensor(),
                tf.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
            model_id = metric.split("_")[-1]

        orig_dset = utils.get_dataset(init_config, tfms)
        orig_dloader = torch.utils.data.DataLoader(
            orig_dset,
            batch_size=config["extraction_batch_sizes"]["metrics"],
            shuffle=False,
            num_workers=config["num_dataloader_workers"],
        )
        orig_labels = torch.tensor(
            orig_dset.label_col.map(lambda x: orig_dset.class_dict[x]).values, dtype=torch.long
        )
        class_names = config["class_names"]
        num_classes = len(class_names)

        # pre-calculate and store features (if applicable)
        if metric.startswith(("mahalanobis", "dreamsim")):
            if metric == "dreamsim":
                metric_layers = None
            else:
                metric_prefix = metric.split(" | ")[0]
                metric_layers = config["metrics"]["layers"][metric_prefix][model_id]
            record_metrics_features(
                config=config,
                timestamp_dir=timestamp_dir,
                model=model,
                transform=tfms,
                metric_layers=metric_layers,
                orig_dloader=orig_dloader,
                model_id=model_id,
            )

        if metric.startswith("lpips"):
            if model_id in ["alex", "vgg", "squeeze"]:
                lpips_model = custom_lpips.LPIPS(model_id, None)
            else:
                lpips_model = custom_lpips.LPIPS(model, config["metrics"]["layers"]["lpips_nolin"][model_id])
            lpips_model.eval().to(config["device"])

        # calculate and store the actual metric
        layer_files = list(timestamp_dir.glob("grid_*.pt"))
        for layer_file in (
            layer_pbar := tqdm(layer_files, total=len(layer_files), desc="metric for vis layer")
        ):
            layer_dict = torch.load(layer_file)
            vis_layer_id = layer_file.stem[5:]
            if (not config["metrics"].get("overwrite", False)) and metric in layer_dict.get(
                "metrics", dict()
            ):
                tqdm.write(f"{metric} already calculated for {vis_layer_id} --> skipped.")
                continue
            layer_pbar.set_postfix_str(vis_layer_id)
            # dist_to_cls: *not nearest neighbor-based* distances to all classes
            # dist_to_imgs: distances to the images in orig_dset --> use for determining nearest neighbors and
            # nn-based class distances
            dist_to_cls, dist_to_imgs = None, None
            if metric.startswith("mahalanobis"):  # <-- does not result in any dist_to_imgs, i.e. no nn-stuff
                metric_layer = config["metrics"]["layers"]["mahalanobis"][model_id][0]
                orig_acts = torch.load(
                    timestamp_dir / "activations4metrics" / model_id / f"{metric_layer}.pt"
                )
                gen_acts = torch.load(
                    timestamp_dir / vis_layer_id / "activations4metrics" / model_id / f"{metric_layer}.pt"
                )
                dist_to_cls, has_invalid_vals = mahalanobis_dist(
                    orig_acts, gen_acts, orig_labels, num_classes
                )
                if has_invalid_vals:
                    tqdm.write(f"{metric}, {metric_layer}: Negative or NaN distance values :(")
            ##
            elif metric.startswith("lpips"):
                gen_dset = datasets._ImageFolderDataset(
                    img_folder=timestamp_dir / vis_layer_id, transform=tf.Compose(tfms)
                )
                gen_dloader = torch.utils.data.DataLoader(
                    gen_dset,
                    batch_size=config["extraction_batch_sizes"]["metrics"],
                    shuffle=False,
                    num_workers=config["num_dataloader_workers"],
                )
                dist_to_cls, dist_to_imgs = lpips_dist(
                    orig_dloader=orig_dloader,
                    gen_dloader=gen_dloader,
                    model_id=model_id,
                    vis_layer_id=vis_layer_id,
                    lpips_model=lpips_model,
                    timestamp_dir=timestamp_dir,
                    device=config["device"],
                    orig_labels=orig_labels,
                    num_classes=num_classes,
                )
            ##
            elif metric == "dreamsim":
                orig_acts = torch.load(timestamp_dir / "activations4metrics" / "dreamsim.pt")
                gen_acts = torch.load(timestamp_dir / vis_layer_id / "activations4metrics" / "dreamsim.pt")
                dist_to_cls, dist_to_imgs = dreamsim_dist(orig_acts, gen_acts, orig_labels, num_classes)

            num_cells = layer_dict["num_cells"]
            if "counts" not in layer_dict:  # == class vis, all indices valid
                valid_ys, valid_xs = torch.nonzero(torch.ones((num_cells, num_cells)), as_tuple=True)
            else:
                valid_ys, valid_xs = torch.nonzero(layer_dict["counts"] > 0, as_tuple=True)
            label_dist_grid = torch.zeros((num_cells, num_cells, num_classes))
            label_dist_grid[valid_ys, valid_xs] = dist_to_cls
            if "metrics" not in layer_dict:
                layer_dict["metrics"] = dict()
            layer_dict["metrics"][metric] = {
                "classes": class_names,
                "counts": label_dist_grid,
                "low_best": True,
            }
            nn_label_counts, nn_label_counts_rel, nn_dists, nn_indices = None, None, None, None
            if dist_to_imgs is not None:
                nn_label_counts, nn_label_counts_rel, nn_dists, nn_indices = nearest_neighbor_stats(
                    dist_to_imgs, int(config["metrics"]["num_nn"]), orig_labels, num_classes
                )
                nn_label_count_grid = torch.zeros((num_cells, num_cells, num_classes), dtype=torch.int)
                nn_label_count_grid[valid_ys, valid_xs] = nn_label_counts
                layer_dict["metrics"][f"{metric}_nn"] = {
                    "classes": class_names,
                    "counts": nn_label_count_grid,
                    "low_best": False,
                }
            # save the results in a tabular format for easier external analysis
            valid_ys = valid_ys.tolist()
            valid_xs = valid_xs.tolist()
            tab_data = {
                "metric": [metric] * len(valid_ys),
                "vis_layer": [vis_layer_id] * len(valid_ys),
                "y_coord": [*valid_ys],
                "x_coord": [*valid_xs],
                "nearest_neighbors": (
                    nn_indices.tolist() if nn_indices is not None else [None] * len(valid_ys)
                ),
                "dists_to_nearest_neighbors": (
                    nn_dists.tolist() if nn_dists is not None else [None] * len(valid_ys)
                ),
                "dists_to_classes": dist_to_cls.tolist(),
                "nearest_neighbors_class_counts": (
                    nn_label_counts.tolist() if nn_label_counts is not None else [None] * len(valid_ys)
                ),
            }
            # also add attribution information to the table export
            if metric_idx == 0 and config["vis_type"] == "atlas":
                attr_name = "avg_attributions_default"
                tab_data["metric"] += [attr_name] * len(valid_ys)
                tab_data["vis_layer"] += [vis_layer_id] * len(valid_ys)
                tab_data["y_coord"] += [*valid_ys]
                tab_data["x_coord"] += [*valid_xs]
                tab_data["nearest_neighbors"] += [None] * len(valid_ys)
                tab_data["dists_to_nearest_neighbors"] += [None] * len(valid_ys)
                # dist isn't really the right notion here, it's rather similarity, but whatever, it's just a naming issue
                tab_data["dists_to_classes"] += layer_dict[attr_name][valid_ys, valid_xs].tolist()
                tab_data["nearest_neighbors_class_counts"] += [None] * len(valid_ys)
            tab_data = pd.DataFrame(tab_data)
            tab_path = timestamp_dir / "metrics.csv"
            tab_data.to_csv(tab_path, index=False, mode="a", header=not tab_path.exists())
            torch.save(layer_dict, layer_file)

        if metric.startswith("lpips"):
            lpips_model.deregister_hooks()

    # clean up any accidental duplicate metric table entries, sort it
    tab_path = timestamp_dir / "metrics.csv"
    tab_data = pd.read_csv(tab_path)
    tab_data = tab_data.drop_duplicates(subset=["metric", "vis_layer", "y_coord", "x_coord"], keep="last")
    tab_data = tab_data.sort_values(
        by=["metric", "vis_layer", "y_coord", "x_coord"],
        key=lambda x: np.argsort(index_natsorted(x)),
        ignore_index=True,
    )
    tab_data.to_csv(tab_path, index=False)

    # retrieve minimal set of metrics from the grid files
    # (the tabular data above may be incomplete due to runs with old code, so maybe this is still
    # somewhat usable in that case)
    extract_grid_metrics(timestamp_dir, config["class_names"])

    cleanup = config["metrics"].get("cleanup", True)
    if cleanup:
        for p in timestamp_dir.glob("**/activations4metrics"):
            shutil.rmtree(str(p))


def extract_grid_metrics(timestamp_dir: Path, class_names: list[str]):
    layer_paths = list(timestamp_dir.glob("grid_*.pt"))
    grid_metrics_path = timestamp_dir / "metrics_from_grid.csv"
    if grid_metrics_path.exists():
        grid_metrics_path.unlink()
    for layer_path in tqdm(
        layer_paths, total=len(layer_paths), desc="Extracting tabular data from grid_*.pt files"
    ):
        layer_dict = torch.load(layer_path)
        vis_layer_id = layer_path.stem[5:]
        num_cells = layer_dict["num_cells"]
        if "counts" not in layer_dict:  # == class vis, all indices valid
            valid_ys, valid_xs = torch.nonzero(torch.ones((num_cells, num_cells)), as_tuple=True)
        else:
            valid_ys, valid_xs = torch.nonzero(layer_dict["counts"] > 0, as_tuple=True)
        metrics_dict = layer_dict.get("metrics", dict())

        base_metric_names = [m for m in metrics_dict if metrics_dict[m]["low_best"]]
        for base_metric in base_metric_names:
            nn_metric = f"{base_metric}_nn"
            if nn_metric in metrics_dict:
                nn_class_counts = metrics_dict[nn_metric]["counts"][valid_ys, valid_xs]
                nearest_class_from_nn_counts = [
                    class_names[idx] for idx in torch.argmax(nn_class_counts, dim=-1)
                ]
                nn_class_counts = nn_class_counts.tolist()
            else:
                nn_class_counts = [None] * len(valid_ys)
                nearest_class_from_nn_counts = [None] * len(valid_ys)
            dists_to_classes = metrics_dict[base_metric]["counts"][valid_ys, valid_xs]
            tab_data = {
                "metric": [base_metric] * len(valid_ys),
                "vis_layer": [vis_layer_id] * len(valid_ys),
                "y_coord": [*(valid_ys.tolist())],
                "x_coord": [*(valid_xs.tolist())],
                "dists_to_classes": dists_to_classes.tolist(),
                "nearest_neighbors_class_counts": nn_class_counts,
                "nearest_class_from_mean": [
                    class_names[idx] for idx in torch.argmin(dists_to_classes, dim=-1)
                ],
                "nearest_class_from_nn_counts": nearest_class_from_nn_counts,
            }
            tab_data = pd.DataFrame(tab_data)
            tab_data.to_csv(grid_metrics_path, index=False, mode="a", header=not grid_metrics_path.exists())

        attr_name = "avg_attributions_default"
        if attr_name in layer_dict:
            attr_values = layer_dict[attr_name][valid_ys, valid_xs]
            tab_data = {
                "metric": [attr_name] * len(valid_ys),
                "vis_layer": [vis_layer_id] * len(valid_ys),
                "y_coord": [*(valid_ys.tolist())],
                "x_coord": [*(valid_xs.tolist())],
                # again: similarity rather than distance, but it's just a naming issue so whatever
                "dists_to_classes": attr_values.tolist(),
                "nearest_neighbors_class_counts": [None] * len(valid_ys),
                "nearest_class_from_mean": [
                    class_names[idx] for idx in torch.argmax(attr_values, dim=-1)
                ],
                "nearest_class_from_nn_counts": [None] * len(valid_ys),
            }
            tab_data = pd.DataFrame(tab_data)
            tab_data.to_csv(
                grid_metrics_path, index=False, mode="a", header=not grid_metrics_path.exists()
            )

        tab_data = pd.read_csv(grid_metrics_path)
        tab_data = tab_data.sort_values(
            by=["metric", "vis_layer", "y_coord", "x_coord"],
            key=lambda x: np.argsort(index_natsorted(x)),
            ignore_index=True,
        )
        tab_data.to_csv(grid_metrics_path, index=False)