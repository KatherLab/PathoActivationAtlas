import torch
import lpips
import torchvision
import inspect
from pathlib import Path


class LPIPS(torch.nn.Module):
    def __init__(self, model, target_layers):
        super().__init__()
        if isinstance(model, str): # model is not actually a model, but a model_id
            model_data = self.get_model_data(model)
            self.net = model_data["model_cls"](weights=model_data["weights"]).features
            self.target_layers = model_data["target_layers"]
            self.lins = torch.nn.ModuleList()
            for i, c in enumerate(model_data["channels"]):
                self.lins.append(lpips.NetLinLayer(c, use_dropout=True))
            weights_path = Path(inspect.getfile(lpips.LPIPS)).parent/f"weights/v0.1/{model}.pth"
            state_dict = torch.load(weights_path, map_location="cpu")
            for i in range(len(self.target_layers)):
                state_dict[f"lins.{i}.model.1.weight"] = state_dict.pop(f"lin{i}.model.1.weight")
            self.load_state_dict(state_dict, strict=False)
        else:
            self.net = model
            self.target_layers = target_layers
            self.lins = None

        self.hook_handles = []
        self.current_activations = dict()
        self.compare_acts = None
        layer_dict = dict([*self.net.named_modules()])
        for layer_name in self.target_layers:
            layer = layer_dict[layer_name]
            self.hook_handles += [layer.register_forward_hook(self.record_activation_hook(layer_name))]

    def __del__(self):
        self.deregister_hooks()

    @staticmethod
    def get_model_data(model_id):
        model_data = {
            "alex": {
                "channels": [64, 192, 384, 256, 256],
                "target_layers": ["1", "4", "7", "9", "11"],
                "model_cls": torchvision.models.alexnet,
                "weights": torchvision.models.AlexNet_Weights.IMAGENET1K_V1
            },
            "vgg": {
                "channels": [64, 128, 256, 512, 512],
                "target_layers": ["3", "8", "15", "22", "29"],
                "model_cls": torchvision.models.vgg16,
                "weights": torchvision.models.VGG16_Weights.IMAGENET1K_V1
            },
            "squeeze": {
                "channels": [64, 128, 256, 384, 384, 512, 512],
                "target_layers": ["1", "4", "7", "9", "10", "11", "12"],
                "model_cls": torchvision.models.squeezenet1_1,
                "weights": torchvision.models.SqueezeNet1_1_Weights.IMAGENET1K_V1
            }
        }
        return model_data[model_id]

    def record_activation_hook(self, layer_name):
        def hook(_, __, output):
            self.current_activations[layer_name] = output.detach()
        return hook

    def deregister_hooks(self):
        """
        Call to restore self.model to unhooked state and clear internal state - basically a clean-up for when
        this object will no longer be used.
        """
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.current_activations = dict()

    def normalize_activations(self, acts, eps=1e-10):
        # normalize activations in the channel dimension or whatever feels closest to that
        if len(acts.shape) in (2,4): # conv feature maps or linear layer-ish output
            channel_dim = 1
        elif len(acts.shape) == 3:  # transformer tokens - tokens correspond to positions, so use the other dimension
            channel_dim = 2
        else:
            raise ValueError("LPIPS_nolin layer output must be shaped like convolutional feature maps (num_dims==4), "
                             "transformer tokens (num_dims==3) or linear layer output (num_dims==2).")
        norm_factor = torch.sqrt(torch.sum(acts ** 2, dim=channel_dim, keepdim=True))
        return acts / (norm_factor+eps)

    def get_activations(self, imgs):
        """
        Returns normalized activations for imgs
        """
        self.net(imgs)
        acts = dict(self.current_activations)
        for layer_name in self.target_layers:
            acts[layer_name] = self.normalize_activations(acts[layer_name])
        return acts

    def get_distances(self, acts_A, acts_B):
        # separating feature extraction from the distance calculation allows for precalculating these features when
        # imgs_A and/or imgs_B need to be repeatedly compared with other imgs
        num_A = len(acts_A[self.target_layers[0]])
        num_B = len(acts_B[self.target_layers[0]])
        dists_per_layer = torch.empty((len(self.target_layers), num_A, num_B))
        for i, layer_name in enumerate(self.target_layers):
            feat_shape = acts_A[layer_name].shape
            # +1 because calculating the difference will result in two "batch" dimensions (one for imgsA, one for imgsB),
            # shifting all other dims back by one
            reduce_dims = [d+1 for d in range(len(feat_shape)) if d != 0]
            spacial_scale = 1.0  # no spacial scaling for linear layers
            if len(feat_shape) == 4:
                spacial_scale = 1 / (feat_shape[2] * feat_shape[3])  # 1 / (height*width)
            elif len(feat_shape) == 3:
                spacial_scale = 1 / feat_shape[1]  # 1 / #tokens
            diffs = (acts_A[layer_name][:, None] - acts_B[layer_name][None, :])**2
            if self.lins is not None:
                # flatten batch dimensions in order to apply lins
                diffs = diffs.reshape((num_A*num_B, *(diffs.shape[2:])))
                diffs = self.lins[i](diffs)
                # .. and undo the flatten
                diffs = diffs.reshape((num_A, num_B, *(diffs.shape[1:])))
            dists_per_layer[i] = spacial_scale * torch.sum(diffs, dim=reduce_dims)
        dists = torch.sum(dists_per_layer, dim=0)
        return dists

    def forward(self, imgs_A, imgs_B):
        acts_A = self.get_activations(imgs_A)
        acts_B = self.get_activations(imgs_B)
        return self.get_distances(acts_A, acts_B)