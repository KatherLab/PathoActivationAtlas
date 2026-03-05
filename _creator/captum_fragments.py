# Code in this file originates from captum's optim-wip branch:
# https://github.com/pytorch/captum/tree/optim-wip


import torch
from torch import nn
from inspect import signature
from typing import Callable, Dict, List, Optional, Tuple, Type, Union, cast
from types import MethodType
from packaging import version
import numpy as np
import requests
from PIL import Image
import math
import matplotlib.pyplot as plt


##################################################################################
##################### stuff needed for NaturalImage ##############################
##################################################################################
TORCH_VERSION = torch.__version__

def make_grid_image(
    tiles: Union[torch.Tensor, List[torch.Tensor]],
    images_per_row: int = 4,
    padding: int = 2,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Make grids from NCHW Image tensors in a way similar to torchvision.utils.make_grid,
    but without any channel duplication or creation behaviour.

    Args:

        tiles (torch.Tensor or list of torch.Tensor): A stack of NCHW image tensors or
            a list of NCHW image tensors to create a grid from.
        nrow (int, optional): The number of rows to use for the grid image.
            Default: 4
        padding (int, optional): The amount of padding between images in the grid
            images.
            padding: 2
        pad_value (float, optional): The value to use for the padding.
            Default: 0.0

    Returns:
        grid_img (torch.Tensor): The full NCHW grid image.
    """
    assert padding >= 0 and images_per_row >= 1
    if isinstance(tiles, (list, tuple)):
        assert all([t.device == tiles[0].device for t in tiles])
        assert all([t.dim() == 4 for t in tiles])
        tiles = torch.cat(tiles, 0)
    assert tiles.dim() == 4

    B, C, H, W = tiles.shape

    x_rows = min(images_per_row, B)
    y_rows = int(math.ceil(float(B) / x_rows))

    base_height = ((H + padding) * y_rows) + padding
    base_width = ((W + padding) * x_rows) + padding

    grid_img = torch.ones(1, C, base_height, base_width, device=tiles.device)
    grid_img = grid_img * pad_value

    n = 0
    for y in range(y_rows):
        for x in range(x_rows):
            if n >= B:
                break
            y_idx = ((H + padding) * y) + padding
            x_idx = ((W + padding) * x) + padding
            grid_img[..., y_idx : y_idx + H, x_idx : x_idx + W] = tiles[n : n + 1]
            n += 1
    return grid_img


def show(
    x: torch.Tensor,
    figsize: Optional[Tuple[int, int]] = None,
    scale: float = 255.0,
    images_per_row: Optional[int] = None,
    padding: int = 2,
    pad_value: float = 0.0,
) -> None:
    """
    Show CHW & NCHW tensors as an image.

    Args:

        x (torch.Tensor): The tensor you want to display as an image.
        figsize (Tuple[int, int], optional): height & width to use
            for displaying the image figure.
        scale (float): Value to multiply the input tensor by so that
            it's value range is [0-255] for display.
        images_per_row (int, optional): The number of images per row to use for the
            grid image. Default is set to None for no grid image creation.
            Default: None
        padding (int, optional): The amount of padding between images in the grid
            images. This parameter only has an effect if nrow is not None.
            Default: 2
        pad_value (float, optional): The value to use for the padding. This parameter
            only has an effect if nrow is not None.
            Default: 0.0
    """

    if x.dim() not in [3, 4]:
        raise ValueError(
            f"Incompatible number of dimensions. x.dim() = {x.dim()}; should be 3 or 4."
        )
    if images_per_row is not None:
        x = make_grid_image(
            x, images_per_row=images_per_row, padding=padding, pad_value=pad_value
        )[0, ...]
    else:
        x = torch.cat([t[0] for t in x.split(1)], dim=2) if x.dim() == 4 else x
    x = x.clone().cpu().detach().permute(1, 2, 0) * scale
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.imshow(x.numpy().astype(np.uint8))
    plt.axis("off")
    plt.show()


def save_tensor_as_image(
    x: torch.Tensor,
    filename: str,
    scale: float = 255.0,
    mode: Optional[str] = None,
    images_per_row: Optional[int] = None,
    padding: int = 2,
    pad_value: float = 0.0,
) -> None:
    """
    Save RGB & RGBA image tensors with a shape of CHW or NCHW as images.

    Args:

        x (torch.Tensor): The tensor you want to save as an image.
        filename (str): The filename to use when saving the image.
        scale (float, optional): Value to multiply the input tensor by so that
            it's value range is [0-255] for saving.
        mode (str, optional): A PIL / Pillow supported colorspace. Default is
            set to None for automatic RGB / RGBA detection and usage.
            Default: None
        images_per_row (int, optional): The number of images per row to use for the
            grid image. Default is set to None for no grid image creation.
            Default: None
        padding (int, optional): The amount of padding between images in the grid
            images. This parameter only has an effect if `nrow` is not None.
            Default: 2
        pad_value (float, optional): The value to use for the padding. This parameter
            only has an effect if `nrow` is not None.
            Default: 0.0
    """

    if x.dim() not in [3, 4]:
        raise ValueError(
            f"Incompatible number of dimensions. x.dim() = {x.dim()}; should be 3 or 4."
        )
    if images_per_row is not None:
        x = make_grid_image(
            x, images_per_row=images_per_row, padding=padding, pad_value=pad_value
        )[0, ...]
    else:
        x = torch.cat([t[0] for t in x.split(1)], dim=2) if x.dim() == 4 else x
    x = x.clone().cpu().detach().permute(1, 2, 0) * scale
    if mode is None:
        mode = "RGB" if x.shape[2] == 3 else "RGBA"
    im = Image.fromarray(x.numpy().astype(np.uint8), mode=mode)
    im.save(filename)


class ImageTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls: Type["ImageTensor"],
        x: Union[List, np.ndarray, torch.Tensor] = [],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:

            x (list or np.ndarray or torch.Tensor): A list, NumPy array, or PyTorch
                tensor to create an `ImageTensor` from.

        Returns:
           x (ImageTensor): An `ImageTensor` instance.
        """
        if isinstance(x, torch.Tensor) and x.is_cuda:
            x.show = MethodType(cls.show, x)
            x.export = MethodType(cls.export, x)
            return x
        else:
            return super().__new__(cls, x, *args, **kwargs)

    @classmethod
    def open(cls, path: str, scale: float = 255.0, mode: str = "RGB") -> "ImageTensor":
        """
        Load an image file from a URL or local filepath directly into an `ImageTensor`.

        Args:

            path (str): A URL or filepath to an image.
            scale (float, optional): The image scale to use.
                Default: 255.0
            mode (str, optional): The image loading mode / colorspace to use.
                Default: "RGB"

        Returns:
           x (ImageTensor): An `ImageTensor` instance.
        """
        if path.startswith("https://") or path.startswith("http://"):
            headers = {"User-Agent": "Captum"}
            response = requests.get(path, stream=True, headers=headers)
            img = Image.open(response.raw)
        else:
            img = Image.open(path)
        img_np = np.array(img.convert(mode)).astype(np.float32)
        return cls(img_np.transpose(2, 0, 1) / scale)

    def __repr__(self) -> str:
        prefix = "ImageTensor("
        indent = len(prefix)
        tensor_str = torch._tensor_str._tensor_str(self, indent)
        suffixes = []
        if self.device.type != torch._C._get_default_device() or (
            self.device.type == "cuda"
            and torch.cuda.current_device() != self.device.index
        ):
            suffixes.append("device='" + str(self.device) + "'")
        return torch._tensor_str._add_suffixes(
            prefix + tensor_str, suffixes, indent, force_newline=self.is_sparse
        )

    @classmethod
    def __torch_function__(
        cls: Type["ImageTensor"],
        func: Callable,
        types: List[Type[torch.Tensor]],
        args: Tuple = (),
        kwargs: dict = None,
    ) -> torch.Tensor:
        if kwargs is None:
            kwargs = {}
        return super().__torch_function__(func, types, args, kwargs)

    def show(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        scale: float = 255.0,
        images_per_row: Optional[int] = None,
        padding: int = 2,
        pad_value: float = 0.0,
    ) -> None:
        """
        Display an `ImageTensor`.

        Args:

            figsize (Tuple[int, int], optional): height & width to use
                for displaying the `ImageTensor` figure.
            scale (float, optional): Value to multiply the `ImageTensor` by so that
                it's value range is [0-255] for display.
                Default: 255.0
            images_per_row (int, optional): The number of images per row to use for the
                grid image. Default is set to None for no grid image creation.
                Default: None
            padding (int, optional): The amount of padding between images in the grid
                images. This parameter only has an effect if `nrow` is not None.
                Default: 2
            pad_value (float, optional): The value to use for the padding. This
                parameter only has an effect if `nrow` is not None.
                Default: 0.0
        """
        show(
            self,
            figsize=figsize,
            scale=scale,
            images_per_row=images_per_row,
            padding=padding,
            pad_value=pad_value,
        )

    def export(
        self,
        filename: str,
        scale: float = 255.0,
        mode: Optional[str] = None,
        images_per_row: Optional[int] = None,
        padding: int = 2,
        pad_value: float = 0.0,
    ) -> None:
        """
        Save an `ImageTensor` as an image file.

        Args:

            filename (str): The filename to use when saving the `ImageTensor` as an
                image file.
            scale (float, optional): Value to multiply the `ImageTensor` by so that
                it's value range is [0-255] for saving.
                Default: 255.0
            mode (str, optional): A PIL / Pillow supported colorspace. Default is
                set to None for automatic RGB / RGBA detection and usage.
                Default: None
            images_per_row (int, optional): The number of images per row to use for the
                grid image. Default is set to None for no grid image creation.
                Default: None
            padding (int, optional): The amount of padding between images in the grid
                images. This parameter only has an effect if `nrow` is not None.
                Default: 2
            pad_value (float, optional): The value to use for the padding. This
                parameter only has an effect if `nrow` is not None.
                Default: 0.0
        """
        save_tensor_as_image(
            self,
            filename=filename,
            scale=scale,
            mode=mode,
            images_per_row=images_per_row,
            padding=padding,
            pad_value=pad_value,
        )


class ToRGB(nn.Module):
    """Transforms arbitrary channels to RGB. We use this to ensure our
    image parametrization itself can be decorrelated. So this goes between
    the image parametrization and the normalization/sigmoid step.
    We offer two precalculated transforms: Karhunen-Loève (KLT) and I1I2I3.
    KLT corresponds to the empirically measured channel correlations on imagenet.
    I1I2I3 corresponds to an approximation for natural images from Ohta et al.[0]
    [0] Y. Ohta, T. Kanade, and T. Sakai, "Color information for region segmentation,"
    Computer Graphics and Image Processing, vol. 13, no. 3, pp. 222–241, 1980
    https://www.sciencedirect.com/science/article/pii/0146664X80900477
    """

    @staticmethod
    def klt_transform() -> torch.Tensor:
        """
        Karhunen-Loève transform (KLT) measured on ImageNet

        Returns:
            **transform** (torch.Tensor): A Karhunen-Loève transform (KLT) measured on
                the ImageNet dataset.
        """
        # Handle older versions of PyTorch
        torch_norm = (
            torch.linalg.norm
            if version.parse(torch.__version__) >= version.parse("1.7.0")
            else torch.norm
        )

        KLT = [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
        transform = torch.Tensor(KLT).float()
        transform = transform / torch.max(torch_norm(transform, dim=0))
        return transform

    @staticmethod
    def i1i2i3_transform() -> torch.Tensor:
        """
        Returns:
            **transform** (torch.Tensor): An approximation of natural colors transform
                (i1i2i3).
        """
        i1i2i3_matrix = [
            [1 / 3, 1 / 3, 1 / 3],
            [1 / 2, 0, -1 / 2],
            [-1 / 4, 1 / 2, -1 / 4],
        ]
        return torch.Tensor(i1i2i3_matrix)

    def __init__(self, transform: Union[str, torch.Tensor] = "klt") -> None:
        """
        Args:

            transform (str or tensor):  Either a string for one of the precalculated
                transform matrices, or a 3x3 matrix for the 3 RGB channels of input
                tensors.
        """
        super().__init__()
        assert isinstance(transform, str) or torch.is_tensor(transform)
        if torch.is_tensor(transform):
            transform = cast(torch.Tensor, transform)
            assert list(transform.shape) == [3, 3]
            self.register_buffer("transform", transform)
        elif transform == "klt":
            self.register_buffer("transform", ToRGB.klt_transform())
        elif transform == "i1i2i3":
            self.register_buffer("transform", ToRGB.i1i2i3_transform())
        else:
            raise ValueError(
                "transform has to be either 'klt', 'i1i2i3'," + " or a matrix tensor."
            )

    @torch.jit.ignore
    def _forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """
        Args:

            x (torch.tensor):  A CHW or NCHW RGB or RGBA image tensor.
            inverse (bool, optional):  Whether to recorrelate or decorrelate colors.
                Default: False.

        Returns:
            chw (torch.tensor):  A tensor with it's colors recorrelated or
                decorrelated.
        """

        assert x.dim() == 3 or x.dim() == 4
        assert x.shape[-3] >= 3
        assert (
            x.names == ("C", "H", "W")
            if x.dim() == 3
            else x.names == ("B", "C", "H", "W")
        )

        # alpha channel is taken off...
        has_alpha = x.size("C") >= 4
        if has_alpha:
            if x.dim() == 3:
                x, alpha_channel = x[:3], x[3:]
            elif x.dim() == 4:
                x, alpha_channel = x[:, :3], x[:, 3:]
            assert x.dim() == alpha_channel.dim()  # ensure we "keep_dim"

        h, w = x.size("H"), x.size("W")
        flat = x.flatten(("H", "W"), "spatials")
        if inverse:
            correct = torch.inverse(self.transform.to(x.device)) @ flat
        else:
            correct = self.transform.to(x.device) @ flat
        chw = correct.unflatten("spatials", (("H", h), ("W", w)))

        if x.dim() == 3:
            chw = chw.refine_names("C", ...)
        elif x.dim() == 4:
            chw = chw.refine_names("B", "C", ...)

        # ...alpha channel is concatenated on again.
        if has_alpha:
            d = 0 if x.dim() == 3 else 1
            chw = torch.cat([chw, alpha_channel], d)

        return chw

    def _forward_without_named_dims(
        self, x: torch.Tensor, inverse: bool = False
    ) -> torch.Tensor:
        """
        JIT compatible forward function for ToRGB.

        Args:

            x (torch.tensor):  A CHW pr NCHW RGB or RGBA image tensor.
            inverse (bool, optional):  Whether to recorrelate or decorrelate colors.
                Default: False.

        Returns:
            chw (torch.tensor):  A tensor with it's colors recorrelated or
                decorrelated.
        """

        assert x.dim() == 4 or x.dim() == 3
        assert x.shape[-3] >= 3

        # alpha channel is taken off...
        has_alpha = x.shape[-3] >= 4
        if has_alpha:
            if x.dim() == 3:
                x, alpha_channel = x[:3], x[3:]
            else:
                x, alpha_channel = x[:, :3], x[:, 3:]
            assert x.dim() == alpha_channel.dim()  # ensure we "keep_dim"
        else:
            # JIT requires a placeholder
            alpha_channel = torch.tensor([0])

        c_dim = 1 if x.dim() == 4 else 0
        h, w = x.shape[c_dim + 1 :]
        flat = x.reshape(list(x.shape[: c_dim + 1]) + [h * w])

        if inverse:
            correct = torch.inverse(self.transform.to(x.device, x.dtype)) @ flat
        else:
            correct = self.transform.to(x.device, x.dtype) @ flat
        chw = correct.reshape(x.shape)

        # ...alpha channel is concatenated on again.
        if has_alpha:
            d = 0 if x.dim() == 3 else 1
            chw = torch.cat([chw, alpha_channel], d)

        return chw

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """
        JIT does not yet support named dimensions.

        Args:

            x (torch.tensor):  A CHW or NCHW RGB or RGBA image tensor.
            inverse (bool, optional):  Whether to recorrelate or decorrelate colors.
                Default: False.

        Returns:
            chw (torch.tensor):  A tensor with it's colors recorrelated or
                decorrelated.
        """
        if torch.jit.is_scripting():
            return self._forward_without_named_dims(x, inverse)
        if list(x.names) in [[None] * 3, [None] * 4]:
            return self._forward_without_named_dims(x, inverse)
        return self._forward(x, inverse)


class InputParameterization(torch.nn.Module):
    def forward(self) -> torch.Tensor:
        raise NotImplementedError


class ImageParameterization(InputParameterization):
    pass


class FFTImage(ImageParameterization):
    """
    Parameterize an image using inverse real 2D FFT
    """

    __constants__ = ["size"]

    def __init__(
        self,
        size: Tuple[int, int] = None,
        channels: int = 3,
        batch: int = 1,
        init: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:

            size (Tuple[int, int]): The height & width dimensions to use for the
                parameterized output image tensor.
            channels (int, optional): The number of channels to use for each image.
                Default: 3
            batch (int, optional): The number of images to stack along the batch
                dimension.
                Default: 1
            init (torch.tensor, optional): Optionally specify a tensor to
                use instead of creating one.
                Default: None
        """
        super().__init__()
        if init is None:
            assert len(size) == 2
            self.size = size
        else:
            assert init.dim() == 3 or init.dim() == 4
            if init.dim() == 3:
                init = init.unsqueeze(0)
            self.size = (init.size(2), init.size(3))
        self.torch_rfft, self.torch_irfft, self.torch_fftfreq = self.get_fft_funcs()

        frequencies = self.rfft2d_freqs(*self.size)
        scale = 1.0 / torch.max(
            frequencies,
            torch.full_like(frequencies, 1.0 / (max(self.size[0], self.size[1]))),
        )
        scale = scale * ((self.size[0] * self.size[1]) ** (1 / 2))
        spectrum_scale = scale[None, :, :, None]

        if init is None:
            coeffs_shape = (
                batch,
                channels,
                self.size[0],
                self.size[1] // 2 + 1,
                2,
            )
            random_coeffs = torch.randn(
                coeffs_shape
            )  # names=["C", "H_f", "W_f", "complex"]
            fourier_coeffs = random_coeffs / 50
        else:
            spectrum_scale = spectrum_scale.to(init.device)
            fourier_coeffs = self.torch_rfft(init) / spectrum_scale

        self.register_buffer("spectrum_scale", spectrum_scale)
        self.fourier_coeffs = nn.Parameter(fourier_coeffs)

    def rfft2d_freqs(self, height: int, width: int) -> torch.Tensor:
        """
        Computes 2D spectrum frequencies.

        Args:

            height (int): The h dimension of the 2d frequency scale.
            width (int): The w dimension of the 2d frequency scale.

        Returns:
            **tensor** (tensor): A 2d frequency scale tensor.
        """

        fy = self.torch_fftfreq(height)[:, None]
        fx = self.torch_fftfreq(width)[: width // 2 + 1]
        return torch.sqrt((fx * fx) + (fy * fy))

    @torch.jit.export
    def torch_irfftn(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.complex64:
            x = torch.view_as_complex(x)
        return torch.fft.irfftn(x, s=self.size)  # type: ignore

    def get_fft_funcs(self) -> Tuple[Callable, Callable, Callable]:
        """
        Support older versions of PyTorch. This function ensures that the same FFT
        operations are carried regardless of whether your PyTorch version has the
        torch.fft update.

        Returns:
            fft functions (tuple of Callable): A list of FFT functions
                to use for irfft, rfft, and fftfreq operations.
        """

        if version.parse(TORCH_VERSION) > version.parse("1.7.0"):
            if version.parse(TORCH_VERSION) <= version.parse("1.8.0"):
                global torch
                import torch.fft

            def torch_rfft(x: torch.Tensor) -> torch.Tensor:
                return torch.view_as_real(torch.fft.rfftn(x, s=self.size))

            torch_irfftn = self.torch_irfftn

            def torch_fftfreq(v: int, d: float = 1.0) -> torch.Tensor:
                return torch.fft.fftfreq(v, d)

        else:

            def torch_rfft(x: torch.Tensor) -> torch.Tensor:
                return torch.rfft(x, signal_ndim=2)

            def torch_irfftn(x: torch.Tensor) -> torch.Tensor:
                return torch.irfft(x, signal_ndim=2)[
                    :, :, : self.size[0], : self.size[1]
                ]

            def torch_fftfreq(v: int, d: float = 1.0) -> torch.Tensor:
                """PyTorch version of np.fft.fftfreq"""
                results = torch.empty(v)
                s = (v - 1) // 2 + 1
                results[:s] = torch.arange(0, s)
                results[s:] = torch.arange(-(v // 2), 0)
                return results * (1.0 / (v * d))

        return torch_rfft, torch_irfftn, torch_fftfreq

    def forward(self) -> torch.Tensor:
        """
        Returns:
            **output** (torch.tensor): A spatially recorrelated tensor.
        """

        scaled_spectrum = self.fourier_coeffs * self.spectrum_scale
        output = self.torch_irfft(scaled_spectrum)
        if torch.jit.is_scripting():
            return output
        return output.refine_names("B", "C", "H", "W")


class NaturalImage(ImageParameterization):
    r"""Outputs an optimizable input image.

    By convention, single images are CHW and float32s in [0,1].
    The underlying parameterization can be decorrelated via a ToRGB transform.
    When used with the (default) FFT parameterization, this results in a fully
    uncorrelated image parameterization. :-)

    If a model requires a normalization step, such as normalizing imagenet RGB values,
    or rescaling to [0,255], it can perform those steps with the provided transforms or
    inside its computation.
    """

    def __init__(
        self,
        size: Tuple[int, int] = (224, 224),
        channels: int = 3,
        batch: int = 1,
        init: Optional[torch.Tensor] = None,
        parameterization: ImageParameterization = FFTImage,
        squash_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        decorrelation_module: Optional[nn.Module] = ToRGB(transform="klt"),
        decorrelate_init: bool = True,
    ) -> None:
        """
        Args:

            size (Tuple[int, int], optional): The height and width to use for the
                nn.Parameter image tensor.
                Default: (224, 224)
            channels (int, optional): The number of channels to use when creating the
                nn.Parameter tensor.
                Default: 3
            batch (int, optional): The number of channels to use when creating the
                nn.Parameter tensor, or stacking init images.
                Default: 1
            parameterization (ImageParameterization, optional): An image
                parameterization class, or instance of an image parameterization class.
                Default: FFTImage
            squash_func (Callable[[torch.Tensor], torch.Tensor]], optional): The squash
                function to use after color recorrelation. A funtion or lambda function.
                Default: None
            decorrelation_module (nn.Module, optional): A ToRGB instance.
                Default: ToRGB
            decorrelate_init (bool, optional): Whether or not to apply color
                decorrelation to the init tensor input.
                Default: True
        """
        super().__init__()
        if not isinstance(parameterization, ImageParameterization):
            # Verify uninitialized class is correct type
            assert issubclass(parameterization, ImageParameterization)
        else:
            assert isinstance(parameterization, ImageParameterization)

        self.decorrelate = decorrelation_module
        if init is not None and not isinstance(parameterization, ImageParameterization):
            assert init.dim() == 3 or init.dim() == 4
            if decorrelate_init and self.decorrelate is not None:
                init = (
                    init.refine_names("B", "C", "H", "W")
                    if init.dim() == 4
                    else init.refine_names("C", "H", "W")
                )
                init = self.decorrelate(init, inverse=True).rename(None)

            if squash_func is None:
                squash_func = self._clamp_image

        self.squash_func = torch.sigmoid if squash_func is None else squash_func
        if not isinstance(parameterization, ImageParameterization):
            parameterization = parameterization(
                size=size, channels=channels, batch=batch, init=init
            )
        self.parameterization = parameterization

    @torch.jit.export
    def _clamp_image(self, x: torch.Tensor) -> torch.Tensor:
        """JIT supported squash function."""
        return x.clamp(0, 1)

    @torch.jit.ignore
    def _to_image_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wrap ImageTensor in torch.jit.ignore for JIT support.

        Args:

            x (torch.tensor): An input tensor.

        Returns:
            x (ImageTensor): An instance of ImageTensor with the input tensor.
        """
        return ImageTensor(x)

    def forward(self) -> torch.Tensor:
        image = self.parameterization()
        if self.decorrelate is not None:
            image = self.decorrelate(image)
        image = image.rename(None)  # TODO: the world is not yet ready
        return self._to_image_tensor(self.squash_func(image))
