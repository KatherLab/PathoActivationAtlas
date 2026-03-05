# grab activations and create atlasses / grids from here
from argparse import ArgumentParser
from pathlib import Path
from _creator import utils, record, atlas, actgrid, class_vis, metrics
from torchvision import transforms as tf

# requires torch v2.0.1 (or lower i guess) or lucent gets sad. see here for downgrading:
# https://pytorch.org/get-started/previous-versions/#v201

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config yaml-file")
    parser.add_argument("--vis_type", type=str, choices=["atlas", "actgrid", "feature_search", "class_vis"],
                        help="currently atlas, actgrid, feature_search or class_vis")
    parser.add_argument("--continue_from", type=str, help="timestamp/folder name of run to continue from (if any)",
                        default=None)
    parser.add_argument("--layout_from", type=str,
                        help="""timestamp/folder name of run to grab layouts from (if they exist). you should only
                        use this when both atlas runs use the same kind of embedding (this will not be checked).
                        Will also use the whitening transform from this run if it exists and if the current run
                        is supposed to be whitened.""",
                        default=None)
    parser.add_argument("--activations_only", action="store_true", help="If set, stop after recording activations.")
    args = parser.parse_args()

    save_dir, config = utils.open_config(args.config, args.vis_type)
    config["continue_from"] = args.continue_from
    config["layout_from"] = args.layout_from
    model = utils.get_model(config)
    tfms, norm = utils.get_transformations(config, model)
    dataset = None

    if config["vis_type"] != "class_vis":
        dataset = utils.get_dataset(config, tfms + [tf.ToTensor()], norm)
        record.record_activations(model, dataset, config, save_dir)
        if args.activations_only:
            print("--activations_only set, exiting now. Bye!")
            exit(0)

    create_funcs = {
        "atlas": atlas.create,
        "actgrid": actgrid.create,
        "class_vis": class_vis.create
    }
    kwargs = {
        "model": model,
        "dataset": dataset,
        "tfms": tfms,
        "norm": norm,
        "config": config,
        "save_dir": save_dir,
        "config_path": Path(args.config)
    }
    if config["vis_type"] in create_funcs:
        timestamp_dir = create_funcs[config["vis_type"]](**kwargs)
        if config["vis_type"] in ["atlas", "class_vis"] and "metrics" in config:
            kwargs = None
            dataset = None
            metrics.calculate_metrics(config, timestamp_dir, model, [*tfms, tf.ToTensor(), norm])