# Class Visualizations and Activation Atlases for Enhancing Interpretability in Deep Learning-Based Computational Pathology

This repository contains the code to reproduce the experiments in the paper.

> [!IMPORTANT]
>
> WIP! For model training, please refer to the `modeling` directory in this repository!

## Setup

Use conda to setup this project's python environment:

```bash
conda env create -f conda_env.yml
conda activate activation-atlas
```

## Usage

Check `config/uni_nct.yaml` for an example of how to set up your configuration file, and `config/NCT-CRC-HE-100K.csv` and `config/CRC-VAL-HE-7K.csv` for examples of the file(s) describing your dataset(s).

### Extract activations and create activation atlases

```bash
python create.py --config path/to/config_file.yaml --vis_type atlas
```

### Create class visualizations

```bash
python create.py --config path/to/config_file.yaml --vis_type class_vis
```

### Interactively explore visualizations and metrics

Start the viewer via `python view.py`. Click on "Open new file", navigate to a previously created `atlas` or `class_vis` file (by default contained at `<save_root>/<experiment name>/<atlas or class_vis>/<timestamp>/`) and open it. You can now:

* Select your layer (for activation atlases) or class (for class visualizations) of interest from the list on the left.
* Pan across (hold left mouse button, move mouse) and zoom into (mouse wheel) the visualization.
* Enable overlays for attributions, ground truth data and metrics from the Overlay menu in the left sidebar.
* View detailed attribution, ground truth and metrics data for individual cells by hovering over them with your mouse.

## External resources

* GUI icons (`_viewer/_resources/*.svg`, `_annotator/_resources/*.svg`) for the viewer and annotator components from [fontawesome.com](https://fontawesome.com/icons).
* Code in `_creator/captum_fragments.py` from [Captum's optim-wip branch](https://github.com/meta-pytorch/captum/tree/optim-wip).