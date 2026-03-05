import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from tqdm import tqdm
import numpy as np
import pprint

from _creator import utils


class BaseAtlasModel(nn.Module):
    def __init__(self, model, layer_name):
        """
        You probably shouldn't explicitly instanciate this class, it's just made to be inherited from.

        model:          Pytorch model
        layer_name:     Name of the layer of interest. Output must be of shape [bs, dim1, dim2, ..., dimn]. For nested
                        layers, start with the topmost module name and separate with dots, e.g.: grandparent.parent.child
        """
        super().__init__()
        self.model = model
        self.layer_name = layer_name
        self._current_activations = None  # activations (all spacial positions) for the most recent forward pass
        self._activations = []  # activation vectors for all forward passes (on cpu)
        self._attributions_default = []  # attribution vectors for all forward passes (on cpu)
        self._input_size = None  # (h, w) shape of input images
        self._act_size = None  # (ch, y, x) shape of the whole activation
        self.positions = None
        self._hook_handles = []
        self._register_layer_hook()

    def _record_activation_hook(self):
        def hook(_, __, output):
            self._current_activations = output
        return hook

    def _register_layer_hook(self):
        layer_dict = dict([*self.model.named_modules()])
        layer = layer_dict[self.layer_name]
        self._hook_handles = []
        self._hook_handles += [layer.register_forward_hook(self._record_activation_hook())]

    def deregister_layer_hook(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def reset_record(self):
        self._current_activations = None  # activations (all spacial positions) for the most recent forward pass
        self._activations = []  # activation vectors for all forward passes (on cpu)
        self._attributions_default = []  # attribution vectors for all forward passes (on cpu)
        self._input_size = None  # (h, w) shape of input images
        self._act_size = None  # (ch, y, x) shape of the whole activation

    def save_record(self, save_path):
        """Concatenate the recorded data into torch.tensors and save it to disk."""
        activations = torch.cat(self._activations, dim=0)
        attributions_default = torch.cat(self._attributions_default, dim=0)
        save_dict = {
            "layer_name": self.layer_name,
            "act": activations,
            "attr_default": attributions_default,
            "input_size": self._input_size,
            "full_act_size": self._act_size,
            "pos_slices": self.positions
        }
        torch.save(save_dict, save_path)


class AtlasConvModel(BaseAtlasModel):
    def __init__(self, model, layer_name, dist_type):
        """
        Wrapper to access spacial layer activations (i.e. the ones for atlas creation) of a pytorch conv layer.

        model:          Pytorch model
        layer_name:     Layer of interest. Output must be of shape [bs, ch, y, x]. For nested layers, start with the
                        topmost module name and separate with dots, e.g.: grandparent.parent.child
        """
        super().__init__(model, layer_name)
        self.dist_type = dist_type
        self.dist_y = None
        self.dist_x = None

    def forward(self, *forward_args, **forward_kwargs):
        output = self.model(*forward_args, **forward_kwargs)
        bs = output.shape[0]
        num_classes = output.shape[1]

        # record input and act size if not already done
        if self._input_size is None:
            inp = forward_args[0].shape
            self._input_size = (inp[2], inp[3])
        if self._act_size is None:
            self._act_size = self._current_activations[0].shape
            if self.dist_type == "uniform":
                self.dist_y = Uniform(1, self._act_size[1]-1) # exclude borders
                self.dist_x = Uniform(1, self._act_size[2]-1)
            elif self.dist_type == "normal":
                mean_y = self._act_size[1] / 2
                mean_x = self._act_size[2] / 2
                std_y = mean_y / 4
                std_x = mean_x / 4
                self.dist_y = Normal(mean_y, std_y)
                self.dist_x = Normal(mean_x, std_x)
            else:
                raise ValueError(f"Invalid conv distribution type {self.dist_type}.")


        # draw random position for each batch element
        y_pos = torch.clamp(self.dist_y.sample((bs,)), min=0, max=self._act_size[1]-1).long()
        x_pos = torch.clamp(self.dist_x.sample((bs,)), min=0, max=self._act_size[2]-1).long()
        batch_dim = torch.arange(0, bs, dtype=torch.long)

        # get gradient d(output)/d(activations). I'm not super sure how to only get the values for the chosen
        # positions rather than everything, so I'll just do it for everything and pick out the values I need.
        num_channels = self._act_size[0]
        dout_dact = torch.zeros((bs, num_classes, num_channels))
        for c in range(num_classes):
            grad = torch.autograd.grad(
                [output[b,c] for b in range(bs)], # each list element is scalar --> no grad_outputs required
                inputs=[self._current_activations],
                retain_graph=True
            )[0]
            dout_dact[:, c] = grad[batch_dim, :, y_pos, x_pos]
        # get attribution vector by multiplying spacial activations with the corresponding gradient element, then
        # averaging across all channels
        self._attributions_default += [
            (dout_dact.detach().cpu() * self._current_activations[batch_dim, :, y_pos, x_pos].detach().cpu()[:, None, :]).mean(dim=2)
        ]

        # record activations
        self._activations += [
            self._current_activations[batch_dim, :, y_pos, x_pos].detach().cpu()    # [bs, ch]
        ]
        return output


class AtlasOtherModel(BaseAtlasModel):
    def __init__(self, model, layer_name, pos_info, actgrid=False):
        """
        Wrapper to access layer activations at specific positions.

        model:          Pytorch model
        layer_name:     Name of the layer of interest. Output must be of shape [bs, dim1, dim2, ..., dimn]. For nested
                        layers, start with the topmost module name and separate with dots, e.g.: grandparent.parent.child
        pos_info:       List/tuple denoting which activations to grab for each of dim1, dim2, ..., dimn
        actgrid:        True if we're extracting activations and attributions for creating an activation grid.
        """
        super().__init__(model, layer_name)
        self.pos_info = pos_info
        self.positions = []
        self.extracted_shape = None # without batch dim
        self.actgrid = actgrid

    def forward(self, *forward_args, **forward_kwargs):
        output = self.model(*forward_args, **forward_kwargs)
        bs = output.shape[0]
        num_classes = output.shape[1]

        # record input and act size, also create the slice objects to access specific positions
        if self._input_size is None:
            self._input_size = (forward_args[0].shape[2], forward_args[0].shape[3])
            self.positions = utils.get_pos_slices(self._current_activations.shape[1:], self.pos_info)
            self.extracted_shape = self._current_activations[0, *self.positions].shape
        if self._act_size is None:
            self._act_size = self._current_activations[0].shape

        # attribution stuff
        dout_dact = torch.zeros((bs, num_classes, *self.extracted_shape))
        for c in range(num_classes):
            grad = torch.autograd.grad(
                [output[b,c] for b in range(bs)],
                inputs=[self._current_activations],
                retain_graph=True
            )[0]
            dout_dact[:, c] = grad[:, *self.positions]
        if self.actgrid:
            self._attributions_default += [
                (dout_dact.detach().cpu() * self._current_activations[:, *self.positions].detach().cpu()[:, None]
                 ).mean(dim=2) # mean across the channel dimension
            ]
        else:
            self._attributions_default += [
                (dout_dact.detach().cpu() * self._current_activations[:, *self.positions].detach().cpu()[:, None] # insert extra dimension for num_classes
                ).view(bs, num_classes, -1).mean(dim=2) # mean across all dimensions except the first two
            ]

        self._activations += [self._current_activations[:, *self.positions].detach().cpu()]
        return output


class FeatureSearchModel(nn.Module):
    def __init__(self, model, layer_name):
        super().__init__()
        self.model = model
        self.layer_name = layer_name
        self._current_activations = None  # activations (all spacial positions) for the most recent forward pass
        self.running_attributions = utils.Welford()
        self._hook_handle = None
        self._register_layer_hook()

    def forward(self, *forward_args, **forward_kwargs):
        output = self.model(*forward_args, **forward_kwargs)
        bs = output.shape[0]
        num_classes = output.shape[1]
        extracted_shape = self._current_activations[0].shape
        dout_dact = torch.zeros((bs, num_classes, *extracted_shape))
        for c in range(num_classes):
            dout_dact[:,c] = torch.autograd.grad(
                [output[b,c] for b in range(bs)],
                inputs=[self._current_activations],
                retain_graph=True
            )[0]
        attributions = dout_dact.detach().cpu() * self._current_activations.detach().cpu()[:, None]
        self.running_attributions.update(attributions)
        return output

    def _record_activation_hook(self):
        def hook(_, __, output):
            self._current_activations = output
        return hook

    def _register_layer_hook(self):
        layer_dict = dict([*self.model.named_modules()])
        layer = layer_dict[self.layer_name]
        self._hook_handle = layer.register_forward_hook(self._record_activation_hook())

    def deregister_layer_hook(self):
        self._hook_handle.remove()
        self._hook_handle = None

    def reset_record(self):
        self._current_activations = None  # activations (all spacial positions) for the most recent forward pass
        self.running_attributions = utils.Welford()

    def save_record(self, save_path):
        mean = self.running_attributions.get_results()["mean"]
        num_classes = mean.shape[0]
        num_dims = len(mean.shape[1:])
        # find slice with highest attribution for each class
        slices = [[] for c in range(num_classes)] # meaning of each dim when done: [class, dim set to 'all', indices for adressing]
        for c in range(num_classes):
            slices[c] = [[] for i in range(num_dims)]
            for i, dim_size in enumerate(mean.shape[1:]):
                dim = i+1
                # mean over dimension of interest
                # (would also happen when calculating the attributions normally)
                mean_mean = mean.mean(dim=dim)[c]
                highest_attr_index = mean_mean.argmax()
                highest_attr_index = np.unravel_index(highest_attr_index, mean_mean.shape)
                slice_info = list(highest_attr_index)
                slice_info = slice_info[:i] + ["all"] + slice_info[i:]
                slices[c][i] = slice_info
        slices = np.array(slices, dtype=object) # np-array to allow better indexing and general numpy shenanigans
        # combine across classes --> reduce class dim, merge class dim into indices dim
        transposed = slices.transpose((1,2,0)) # [all, indices, class] --> want [all, indices] (where indices can be "all" or a list) by reducing class
        slices_combined_classes = utils.merge_dim_output_use_all_ufunc.reduce(transposed, axis=2)
        # combine across classes and dimensions --> reduce class- and dim set to 'all'-dims
        # also, throw away "all"-entries since every index would just result in "all" here otherwise.
        transposed = slices.transpose((2,0,1)) # [indices, class, all]
        slices_combined_classes_dims = utils.merge_dim_output_ufunc.reduce(
            utils.merge_dim_output_ufunc.reduce(transposed, axis=2), axis=1
        )

        save_dict = {
            "attributions": self.running_attributions.get_results(),
            "slices": slices,
            "slices_combined_classes": slices_combined_classes,
            "slices_combined_classes_dims": slices_combined_classes_dims
        }
        save_path_txt = save_path.parent/f"{save_path.stem}.txt"
        with save_path_txt.open("a") as f:
            print(f"---------------- Slice dicts for {self.layer_name}: ----------------", file=f)
            print("----- SLICES:", file=f)
            print(f"{pprint.pformat(save_dict['slices'], width=200)}", file=f)
            print("----- SLICES_COMBINED_CLASSES:", file=f)
            print(f"{pprint.pformat(save_dict['slices_combined_classes'], width=200)}", file=f)
            print("----- SLICES_COMBINED_CLASSES_DIMS:", file=f)
            print(f"{pprint.pformat(save_dict['slices_combined_classes_dims'], width=200)}", file=f)
            print("---------------------------------------------------------------------------\n", file=f)
        torch.save(save_dict, save_path)


class ActivationOnlyModel(nn.Module):
    """
    Model wrapper for recording activations at specific positions, minus all the attributions stuff of
    BaseAtlasModel children
    """
    def __init__(self, model, layer_name, pos_info):
        super().__init__()
        self.model = model
        self.layer_name = layer_name
        self._current_activations = None
        self._activations = []
        self._input_size = None
        self._act_size = None
        self.pos_info = pos_info
        self.positions = []
        self.extracted_shape = None # no batch dim
        self._register_layer_hook()

    def forward(self, *forward_args, **forward_kwargs):
        output = self.model(*forward_args, **forward_kwargs)
        # record input and act size, also create the slice objects to access specific positions
        if self._input_size is None:
            self._input_size = (forward_args[0].shape[2], forward_args[0].shape[3])
            self.positions = utils.get_pos_slices(self._current_activations.shape[1:], self.pos_info)
            self.extracted_shape = self._current_activations[0, *self.positions].shape
        if self._act_size is None:
            self._act_size = self._current_activations[0].shape
        self._activations += [self._current_activations[:, *self.positions].detach().cpu()]
        return output

    def _record_activation_hook(self):
        def hook(_, __, output):
            self._current_activations = output
        return hook

    def _register_layer_hook(self):
        layer_dict = dict([*self.model.named_modules()])
        layer = layer_dict[self.layer_name]
        self._hook_handles = []
        self._hook_handles += [layer.register_forward_hook(self._record_activation_hook())]

    def deregister_layer_hook(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def reset_record(self):
        self._current_activations = None  # activations (all spacial positions) for the most recent forward pass
        self._activations = []  # activation vectors for all forward passes (on cpu)
        self._input_size = None  # (h, w) shape of input images
        self._act_size = None  # (ch, y, x) shape of the whole activation
        self.positions = []
        self.extracted_shape = None

    def get_record(self, save_path=None):
        activations = torch.cat(self._activations, dim=0)
        save_dict = {
            "layer_name": self.layer_name,
            "act": activations,
            "input_size": self._input_size,
            "full_act_size": self._act_size,
            "pos_slices": self.positions
        }
        if save_path is not None:
            torch.save(save_dict, save_path)
        return save_dict

def record_activations(model, dataset, config, save_dir):
    vis_type = config["vis_type"]
    device = config["device"]
    dloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["extraction_batch_sizes"][vis_type],
        shuffle=False,
        num_workers=config["num_dataloader_workers"]
    )
    layers = config["layers"]
    conv_activation_distribution = config.get("conv_activation_distribution", "uniform")
    extra_forward_kwargs = config["model"]["extra_forward_kwargs"]
    if extra_forward_kwargs is None:
        extra_forward_kwargs = dict()
    extra_forward_args = config["model"]["extra_forward_args"]
    if extra_forward_args is None:
        extra_forward_args = []

    for layer_name in tqdm(layers, total=len(layers), desc=f"Creating {vis_type} activations..."):
        if vis_type == "feature_search":
            pos_infos = [None]
        else:
            pos_infos = layers[layer_name]
        for pos_info in pos_infos:
            if vis_type == "actgrid" and pos_info != "conv":
                continue
            hooked_model_info = {
                "atlas": {
                    "class": AtlasConvModel if pos_info == "conv" else AtlasOtherModel,
                    "args": [
                        model, layer_name, conv_activation_distribution
                    ] if pos_info == "conv" else [
                        model, layer_name, pos_info, False
                    ]
                },
                "actgrid": {
                    "class": AtlasOtherModel,
                    "args": [model, layer_name, ("all", "all", "all"), True]
                },
                "feature_search": {
                    "class": FeatureSearchModel,
                    "args": [model, layer_name]
                }
            }
            pos_str = "" if pos_info == "conv" or pos_info is None else f" {utils.get_pos_string(pos_info)}"
            file_path = save_dir/f"{layer_name}{pos_str}.pt"
            if not file_path.exists():
                hooked_model = hooked_model_info[vis_type]["class"](*hooked_model_info[vis_type]["args"])
                hooked_model.eval()
                for imgs, _ in tqdm(dloader, total=len(dloader), desc=f"{layer_name}{pos_str}, processing batches...", leave=False):
                    imgs = imgs.to(device)
                    hooked_model(imgs, *extra_forward_args, **extra_forward_kwargs)
                hooked_model.save_record(file_path)
                hooked_model.deregister_layer_hook()