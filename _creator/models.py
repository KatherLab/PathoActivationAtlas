import torch
import timm
import torch.nn as nn
import torchvision
from collections import OrderedDict
from pathlib import Path
from torchvision import transforms as tf
from timm.layers import SwiGLUPacked


class Inceptionv1(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.model = torchvision.models.googlenet(pretrained=pretrained)
        num_fc_in = self.model.fc.in_features
        self.model.fc = nn.Linear(num_fc_in, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNet(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.resnet = model
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)


class ResNet18(ResNet):
    def __init__(self, num_classes, pretrained=True):
        super().__init__(torchvision.models.resnet18(pretrained=pretrained), num_classes)


# Architecture modified from https://keras.io/examples/vision/mnist_convnet/
class ColoredMNISTNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, 32, kernel_size=3)),
            ("relu0", nn.ReLU(inplace=True)),
            ("maxpool0", nn.MaxPool2d(kernel_size=2, stride=2)),
            ("conv1", nn.Conv2d(32, 64, kernel_size=3)),
            ("relu1", nn.ReLU(inplace=True)),
            ("maxpool1", nn.AdaptiveMaxPool2d((5, 5))) # [bs, 64, 5, 5] --> [bs, 1600] after flattening
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ("flatten", nn.Flatten()),
            ("dropout", nn.Dropout(p=0.5)),
            ("fc", nn.Linear(1600, num_classes))
        ]))

    def forward(self, x):
        return self.classifier(self.features(x))


class ConvStem(nn.Module):
    """Custom Patch Embed Layer, from https://huggingface.co/1aurent/swin_tiny_patch4_window7_224.CTransPath
    Adapted from https://github.com/Xiyue-Wang/TransPath/blob/main/ctran.py#L6-L44
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, **kwargs):
        super().__init__()

        # Check input constraints
        assert patch_size == 4, "Patch size must be 4"
        assert embed_dim % 8 == 0, "Embedding dimension must be a multiple of 8"

        img_size = timm.layers.helpers.to_2tuple(img_size)
        patch_size = timm.layers.helpers.to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Create stem network
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        # Apply normalization layer (if provided)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        # Check input image size
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        x = self.norm(x)
        return x


class CTransPath(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(
            model_name="hf-hub:1aurent/swin_tiny_patch4_window7_224.CTransPath",
            embed_layer=ConvStem,
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


class VisionTransformerB16(nn.Module):
    def __init__(self, num_classes, weights="IMAGENET1K_V1"):
        super().__init__()
        self.num_classes = num_classes
        self.model = torchvision.models.vit_b_16(weights=weights)
        self.model.heads = nn.Linear(768, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)


class ConvNext(nn.Module):
    def __init__(self, num_classes, weights="convnextv2_tiny.fcmae"):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(
            model_name=weights, pretrained=True, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


class UNI(nn.Module):
    def __init__(self, num_classes, model_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=num_classes,
            dynamic_img_size=True
        )
        # strict=False because the classifier head will not match
        if model_path is not None:
            self.model.load_state_dict(torch.load(Path(model_path), map_location="cpu"), strict=False)

    def forward(self, x):
        return self.model(x)

    def get_transforms(self):
        tfms = [tf.Resize(224)]
        norm = tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return tfms, norm


########### the following models don't have a classification layer, so they're not suitable for atlas/class_vis/actgrid
########### as they are. feel free to implement a wrapper with a classifier though.

# usage requires requesting access at https://huggingface.co/prov-gigapath/prov-gigapath and then
# exporting your read access token: export HF_TOKEN=<huggingface read-only token> (if that doesn't work, try logging
# in with your token via huggingface-cli login)
class ProvGigaPath(nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        self.model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=model_path is None)
        if model_path is not None:
            self.model.load_state_dict(torch.load(Path(model_path), map_location="cpu"), strict=False)

    def forward(self, x):
        return self.model(x)

    def get_transforms(self):
        tfms = [
            tf.Resize(256, interpolation=tf.InterpolationMode.BICUBIC),
            tf.CenterCrop(224)
        ]
        norm = tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return tfms, norm


class HOptimus0(nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        self.model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0", pretrained=model_path is None, init_values=1e-5, dynamic_img_size=False
        )
        if model_path is not None:
            self.model.load_state_dict(torch.load(Path(model_path), map_location="cpu"), strict=False)

    def forward(self, x):
        return self.model(x)

    def get_transforms(self):
        # from hf: "H-optimus-0 expects images of size 224x224 that were extracted at 0.5 microns per pixel."
        # In case that's not available, let's just resize to 224 anyways.
        tfms = [tf.Resize(224)]
        norm = tf.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517))
        return tfms, norm


# usage requires requesting access at https://huggingface.co/paige-ai/Virchow2 and then
# exporting your read access token: export HF_TOKEN=<huggingface read-only token> (if that doesn't work, try logging
# in with your token via huggingface-cli login)
class Virchow2(nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        self.model = timm.create_model(
            "hf-hub:paige-ai/Virchow2", pretrained=model_path is None, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU
        )
        if model_path is not None:
            self.model.load_state_dict(torch.load(Path(model_path), map_location="cpu"), strict=False)
        self.embhead = torch.nn.Identity()

    def forward(self, x, retval="emb"):
        """
        retval options:
        - "emb" (or arbitrary value) for class token concatenated with mean of patch tokens
        - "full" for all tokens
        - "class" for class token only
        """
        out = self.model(x)
        if retval == "full":
            return self.embhead(out)
        class_token = out[:, 0]
        if retval == "class":
            return self.embhead(class_token)
        patch_tokens = out[:, 5:]
        emb = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        return self.embhead(emb)

    def get_transforms(self):
        tfms = [tf.Resize(224, interpolation=tf.InterpolationMode.BICUBIC)]
        norm = tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return tfms, norm
