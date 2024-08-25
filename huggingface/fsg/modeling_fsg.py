import math
import re
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from transformers import PreTrainedModel
from huggingface_hub import PyTorchModelHubMixin
from numba import jit
from .configuration import FsgConfig
from typing import Literal, Type, Union, List


def batch_fn(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def gaussian_kernel_1d(sigma: float, num_sigmas: float = 3.0) -> torch.Tensor:
    radius = math.ceil(num_sigmas * sigma)
    support = torch.arange(-radius, radius + 1, dtype=torch.float)
    kernel = Normal(loc=0, scale=sigma).log_prob(support).exp_()
    # Ensure kernel weights sum to 1, so that image brightness is not altered
    return kernel.mul_(1 / kernel.sum())


def gaussian_filter_2d(img: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel_1d = gaussian_kernel_1d(sigma).to(img.device)  # Create 1D Gaussian kernel
    padding = len(kernel_1d) // 2  # Ensure that image size does not change
    img = img[None, None, ...]  # Need 4D data for ``conv2d()``
    # Convolve along columns and rows
    img = F.conv2d(img, weight=kernel_1d.view(1, 1, -1, 1), padding=(padding, 0))
    img = F.conv2d(img, weight=kernel_1d.view(1, 1, 1, -1), padding=(0, padding))
    return img.squeeze()  # Make 2D again


class BaseModel(nn.Module):
    def __init__(
        self,
        patch_size: int,
        num_classes: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes


class ConstrainedConv(nn.Module):
    def __init__(self, input_chan=3, num_filters=6, is_constrained=True):
        super().__init__()
        self.kernel_size = 5
        self.input_chan = input_chan
        self.num_filters = num_filters
        self.is_constrained = is_constrained
        weight = torch.empty(num_filters, input_chan, self.kernel_size, self.kernel_size)
        nn.init.xavier_normal_(weight, gain=1 / 3)
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.one_middle = torch.zeros(self.kernel_size * self.kernel_size)
        self.one_middle[12] = 1
        self.one_middle = nn.Parameter(self.one_middle, requires_grad=False)

    def forward(self, x):
        w = self.weight
        if self.is_constrained:
            w = w.view(-1, self.kernel_size * self.kernel_size)
            w = w - w.mean(1)[..., None] + 1 / (self.kernel_size * self.kernel_size - 1)
            w = w - (w + 1) * self.one_middle
            w = w.view(self.num_filters, self.input_chan, self.kernel_size, self.kernel_size)
        x = nn.functional.conv2d(x, w, padding="valid")
        x = nn.functional.pad(x, (2, 3, 2, 3))
        return x


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_chans,
        out_chans,
        kernel_size,
        stride,
        padding,
        activation: Literal["tanh", "relu"],
    ):
        super().__init__()
        assert activation.lower() in ["tanh", "relu"], "The activation layer must be either Tanh or ReLU"
        self.conv = torch.nn.Conv2d(
            in_chans,
            out_chans,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = torch.nn.BatchNorm2d(out_chans)
        self.act = torch.nn.Tanh() if activation.lower() == "tanh" else torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2)

    def forward(self, x):
        return self.maxpool(self.act(self.bn(self.conv(x))))


class DenseBlock(torch.nn.Module):
    def __init__(
        self,
        in_chans,
        out_chans,
        activation: Literal["tanh", "relu"],
    ):
        super().__init__()
        assert activation.lower() in ["tanh", "relu"], "The activation layer must be either Tanh or ReLU"
        self.fc = torch.nn.Linear(in_chans, out_chans)
        self.act = torch.nn.Tanh() if activation.lower() == "tanh" else torch.nn.ReLU()

    def forward(self, x):
        return self.act(self.fc(x))


class MISLNet(BaseModel):
    arch = {
        "p256": [
            ("conv1", -1, 96, 7, 2, "valid", "tanh"),
            ("conv2", 96, 64, 5, 1, "same", "tanh"),
            ("conv3", 64, 64, 5, 1, "same", "tanh"),
            ("conv4", 64, 128, 1, 1, "same", "tanh"),
            ("fc1", 6 * 6 * 128, 200, "tanh"),
            ("fc2", 200, 200, "tanh"),
        ],
        "p256_3fc_256e": [
            ("conv1", -1, 96, 7, 2, "valid", "tanh"),
            ("conv2", 96, 64, 5, 1, "same", "tanh"),
            ("conv3", 64, 64, 5, 1, "same", "tanh"),
            ("conv4", 64, 128, 1, 1, "same", "tanh"),
            ("fc1", 6 * 6 * 128, 1024, "tanh"),
            ("fc2", 1024, 512, "tanh"),
            ("fc3", 512, 256, "tanh"),
        ],
        "p128": [
            ("conv1", -1, 96, 7, 2, "valid", "tanh"),
            ("conv2", 96, 64, 5, 1, "same", "tanh"),
            ("conv3", 64, 64, 5, 1, "same", "tanh"),
            ("conv4", 64, 128, 1, 1, "same", "tanh"),
            ("fc1", 2 * 2 * 128, 200, "tanh"),
            ("fc2", 200, 200, "tanh"),
        ],
        "p96": [
            ("conv1", -1, 96, 7, 2, "valid", "tanh"),
            ("conv2", 96, 64, 5, 1, "same", "tanh"),
            ("conv3", 64, 64, 5, 1, "same", "tanh"),
            ("conv4", 64, 128, 1, 1, "same", "tanh"),
            ("fc1", 8 * 4 * 64, 200, "tanh"),
            ("fc2", 200, 200, "tanh"),
        ],
        "p64": [
            ("conv1", -1, 96, 7, 2, "valid", "tanh"),
            ("conv2", 96, 64, 5, 1, "same", "tanh"),
            ("conv3", 64, 64, 5, 1, "same", "tanh"),
            ("conv4", 64, 128, 1, 1, "same", "tanh"),
            ("fc1", 2 * 4 * 64, 200, "tanh"),
            ("fc2", 200, 200, "tanh"),
        ],
    }

    def __init__(
        self,
        patch_size: int,
        variant: str,
        num_classes=0,
        num_filters=6,
        is_constrained=True,
        **kwargs,
    ):
        super().__init__(patch_size, num_classes)
        self.variant = variant
        self.chosen_arch = self.arch[variant]
        self.num_filters = num_filters

        self.constrained_conv = ConstrainedConv(num_filters=num_filters, is_constrained=is_constrained)

        self.conv_blocks = []
        self.fc_blocks = []
        for block in self.chosen_arch:
            if block[0].startswith("conv"):
                self.conv_blocks.append(
                    ConvBlock(
                        in_chans=(num_filters if block[1] == -1 else block[1]),
                        out_chans=block[2],
                        kernel_size=block[3],
                        stride=block[4],
                        padding=block[5],
                        activation=block[6],
                    )
                )
            elif block[0].startswith("fc"):
                self.fc_blocks.append(
                    DenseBlock(
                        in_chans=block[1],
                        out_chans=block[2],
                        activation=block[3],
                    )
                )

        self.conv_blocks = nn.Sequential(*self.conv_blocks)
        self.fc_blocks = nn.Sequential(*self.fc_blocks)

        self.register_buffer("flatten_index_permutation", torch.tensor([0, 1, 2, 3], dtype=torch.long))

        if self.num_classes > 0:
            self.output = nn.Linear(self.chosen_arch[-1][2], self.num_classes)

    def forward(self, x):
        x = self.constrained_conv(x)
        x = self.conv_blocks(x)
        x = x.permute(*self.flatten_index_permutation)
        x = x.flatten(1, -1)
        x = self.fc_blocks(x)
        if self.num_classes > 0:
            x = self.output(x)
        return x

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if "flatten_index_permutation" not in state_dict:
            super().load_state_dict(state_dict, False, assign)
        else:
            super().load_state_dict(state_dict, strict, assign)


class CompareNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, output_dim=64):
        super().__init__()
        self.fc1 = DenseBlock(input_dim, hidden_dim, "relu")
        self.fc2 = DenseBlock(hidden_dim * 3, output_dim, "relu")
        self.fc3 = nn.Linear(output_dim, 2)

    def forward(self, x1, x2):
        x1 = self.fc1(x1)
        x2 = self.fc1(x2)
        x = torch.cat((x1, x1 * x2, x2), dim=1)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class FSM(nn.Module):
    """
    FSM (Forensic Similarity Metric) is a neural network module that computes the similarity between two input images using a feature extraction module and a comparison network module.

    Args:
        fe_config (dict): Configuration for the feature extraction module.
        comparenet_config (dict): Configuration for the comparison network module.
        fe_ckpt (str): Path to the checkpoint file for the feature extraction module.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        fe_config,
        comparenet_config,
        fe_ckpt=None,
        **kwargs,
    ):
        super().__init__()
        fe_config["num_classes"] = 0  # to make fe without final classification layer
        self.fe: MISLNet = self.load_module_from_ckpt(MISLNet, fe_ckpt, "", **fe_config)
        self.patch_size = self.fe.patch_size
        comparenet_config["input_dim"] = self.fe.fc_blocks[-1].fc.out_features
        self.comparenet = CompareNet(**comparenet_config)
        self.fe_freeze = True

    def load_module_state_dict(self, module: nn.Module, state_dict, module_name=""):
        curr_model_state_dict = module.state_dict()
        curr_model_keys_status = {k: False for k in curr_model_state_dict.keys()}
        outstanding_keys = []
        for ckpt_layer_name, ckpt_layer_weights in state_dict.items():
            if module_name not in ckpt_layer_name:
                continue
            ckpt_matches = re.findall(r"(?=(?:^|\.)((?:\w+\.)*\w+)$)", ckpt_layer_name)[::-1]
            model_layer_name_match = list(set(ckpt_matches).intersection(set(curr_model_state_dict.keys())))
            # print(ckpt_layer_name, model_layer_name_match)
            if len(model_layer_name_match) == 0:
                outstanding_keys.append(ckpt_layer_name)
            else:
                model_layer_name = model_layer_name_match[0]
                assert (
                    curr_model_state_dict[model_layer_name].shape == ckpt_layer_weights.shape
                ), f"Ckpt layer '{ckpt_layer_name}' shape {ckpt_layer_weights.shape} does not match model layer '{model_layer_name}' shape {curr_model_state_dict[model_layer_name].shape}"
                curr_model_state_dict[model_layer_name] = ckpt_layer_weights
                curr_model_keys_status[model_layer_name] = True

        if all(curr_model_keys_status.values()):
            print(f"Success! All necessary keys for module '{module.__class__.__name__}' are loaded!")
        else:
            not_loaded_keys = [k for k, v in curr_model_keys_status.items() if not v]
            print(f"Warning! Some keys are not loaded! Not loaded keys are:\n{not_loaded_keys}")
            if len(outstanding_keys) > 0:
                print(f"Outstanding keys are: {outstanding_keys}")
        module.load_state_dict(curr_model_state_dict, strict=False)

    def load_module_from_ckpt(
        self,
        module_class: Type[nn.Module],
        ckpt_path: Union[None, str],
        module_name: str,
        *args,
        **kwargs,
    ) -> nn.Module:
        module = module_class(*args, **kwargs)

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            ckpt_state_dict = ckpt["state_dict"]
            self.load_module_state_dict(module, ckpt_state_dict, module_name=module_name)
        return module

    def load_state_dict(self, state_dict, strict=True, assign=False):
        try:
            super().load_state_dict(state_dict, strict=strict, assign=assign)
        except Exception as e:
            print(f"Error loading state dict using normal method: {e}")
            print("Trying to load state dict manually...")
            # self.load_module_state_dict(self.fe, state_dict, module_name="fe")
            # self.load_module_state_dict(self.comparenet, state_dict, module_name="comparenet")
            self.load_module_state_dict(self, state_dict, module_name="")
            print("State dict loaded successfully!")

    def forward_fe(self, x):
        if self.freeze_fe:
            self.fe.eval()
            with torch.no_grad():
                return self.fe(x)
        else:
            self.fe.train()
            return self.fe(x)

    def forward(self, x1, x2):
        x1 = self.forward_fe(x1)
        x2 = self.forward_fe(x2)
        return self.comparenet(x1, x2)


class FsgModel(
    PreTrainedModel,
    PyTorchModelHubMixin,
    repo_url="ductai199x/forensic-similarity-graph",
    pipeline_tag="image-manipulation-detection-localization",
    license="cc-by-nc-nd-4.0",
):
    """
    Forensic Similarity Graph (FSG) algorithm.
    https://ieeexplore.ieee.org/abstract/document/9113265

    This class is designed to create a graph-based representation of forensic similarity between different patches of an image, allowing for the detection of manipulated regions.

    Args:
        stride_ratio (float): The ratio of the stride to the patch size, determining the overlap between patches. The lower the value, the higher the overlap.
        fast_sim_mode (bool): If True, the algorithm uses a faster method to compute similarity scores, potentially at the cost of accuracy.
        loc_threshold (float): The threshold for determining the location of interest in the similarity graph. Values above this threshold are considered significant.
        is_high_sim (bool): If True, higher similarity scores indicate higher similarity. If False, lower scores indicate higher similarity.
        need_input_255 (bool): If True, input images are expected to be scaled to [0, 255]. If False, images are expected to be in [0, 1].
        **kwargs: Additional keyword arguments passed to the superclass initializer.

    Example Usage:
    ```python
    import torch
    import matplotlib.pyplot as plt
    from torchvision.io import read_image, ImageReadMode
    from model import FSG

    ckpt_path = "path/to/ckpt.pth"
    model = FSG.load_from_checkpoint(ckpt_path, map_location="cpu", stride_ratio=0.5, fast_sim_mode=False, loc_threshold=0.37, is_high_sim=False, need_input_255=False)
    model.eval()

    img_path = "path/to/image.jpg"
    image = read_image(img_path, mode=ImageReadMode.RGB).float() / 255

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        img_preds, loc_preds = model(image[None, ...].to(device))

    plt.imshow(loc_preds.cpu()[0])
    plt.colorbar()
    plt.show()
    ```
    """

    config_class = FsgConfig

    def __init__(self, config: FsgConfig, **kwargs):
        super().__init__(config)
        self.patch_size = config.fe_config.patch_size
        self.stride = int(self.patch_size * config.stride_ratio)
        self.fast_sim_mode = config.fast_sim_mode
        self.loc_threshold = config.loc_threshold
        self.is_high_sim = True
        self.need_input_255 = config.need_input_255
        self.model = FSM(
            fe_config=config.fe_config.to_dict(), comparenet_config=config.comparenet_config.to_dict()
        )

        warnings.filterwarnings("ignore")

    def get_batched_patches(self, x: torch.Tensor):
        B, C, H, W = x.shape
        # split images into batches of patches: B x C x H x W -> B x (NumPatchHeight x NumPatchWidth) x C x PatchSize x PatchSize
        batched_patches = (
            x.unfold(2, self.patch_size, self.stride)
            .unfold(3, self.patch_size, self.stride)
            .permute(0, 2, 3, 1, 4, 5)
        )
        batched_patches = batched_patches.contiguous().view(B, -1, C, self.patch_size, self.patch_size)
        return batched_patches

    def get_patches_single(self, x: torch.Tensor):
        C, H, W = x.shape
        patches = (
            x.unfold(1, self.patch_size, self.stride)
            .unfold(2, self.patch_size, self.stride)
            .permute(1, 2, 0, 3, 4)
        )
        patches = patches.contiguous().view(-1, C, self.patch_size, self.patch_size)
        return patches

    @jit(forceobj=True)
    def get_features(self, image_patches: torch.Tensor):
        patches_features = []
        for batch in list(batch_fn(image_patches, 256)):
            batch = batch.float()
            feats = self.model.fe(batch).detach()
            patches_features.append(feats)
        patches_features = torch.vstack(patches_features)
        return patches_features

    @jit(forceobj=True)
    def get_sim_scores(self, patch_pairs):
        patches_sim_scores = []
        for batch in list(batch_fn(patch_pairs, 4096)):
            batch = batch.permute(1, 0, 2).float()
            scores = self.model.comparenet(*batch).detach()
            scores = torch.nn.functional.softmax(scores, dim=1)
            patches_sim_scores.append(scores)
        patches_sim_scores = torch.vstack(patches_sim_scores)
        return patches_sim_scores

    def forward_single(self, patches: torch.Tensor):
        P, C, H, W = patches.shape
        features = self.get_features(patches)
        sim_mat = torch.zeros(P, P, device=patches.device)
        if self.fast_sim_mode:
            upper_tri_idx = torch.triu_indices(P, P, 1).T
            patch_pairs = features[upper_tri_idx]
        else:
            patch_cart_prod = torch.cartesian_prod(torch.arange(P), torch.arange(P))
            patch_pairs = features[patch_cart_prod]
        sim_scores = self.get_sim_scores(patch_pairs).detach()
        if self.fast_sim_mode:
            sim_mat[upper_tri_idx[:, 0], upper_tri_idx[:, 1]] = sim_scores[:, 1]
            sim_mat += sim_mat.clone().T
        else:
            sim_mat = sim_scores[:, 1].view(P, P)
            sim_mat = 0.5 * (sim_mat + sim_mat.T)
        if not self.is_high_sim:
            sim_mat = 1 - sim_mat
        sim_mat.fill_diagonal_(0.0)
        degree_mat = torch.diag(sim_mat.sum(axis=1))
        laplacian_mat = degree_mat - sim_mat
        degree_sym_mat = torch.diag(sim_mat.sum(axis=1) ** -0.5)
        laplacian_sym_mat = (degree_sym_mat @ laplacian_mat) @ degree_sym_mat
        eigvals, eigvecs = torch.linalg.eigh(laplacian_sym_mat.cpu())
        spectral_gap = eigvals[1] - eigvals[0]
        img_pred = 1 - spectral_gap
        eigvec = eigvecs[:, 1]
        patch_pred = (eigvec > 0).int()
        return img_pred.detach(), patch_pred.detach()

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(x, torch.Tensor) and len(x.shape) == 3:
            x = [x]

        img_preds = []
        loc_preds = []
        for img in x:
            C, H, W = img.shape
            if self.need_input_255 and img.max() <= 1:
                img = img * 255
            # get the (x, y) coordinates of the top left of each patch in the image
            x_inds = torch.arange(W).unfold(0, self.patch_size, self.stride)[:, 0]
            y_inds = torch.arange(H).unfold(0, self.patch_size, self.stride)[:, 0]
            xy_inds = torch.tensor([(ii, jj) for jj in y_inds for ii in x_inds]).to(img.device)

            patches = self.get_patches_single(img)
            img_pred, patch_pred = self.forward_single(patches)
            loc_pred = self.patch_to_pixel_pred(patch_pred, xy_inds)
            loc_pred = F.interpolate(loc_pred[None, None, ...], size=(H, W), mode="nearest").squeeze()
            img_preds.append(img_pred)
            loc_preds.append(loc_pred)
        return img_preds, loc_preds

    def patch_to_pixel_pred(self, patch_pred, xy_inds):
        W, H = torch.max(xy_inds, dim=0).values + self.patch_size
        pixel_pred = torch.zeros((H, W)).to(patch_pred.device)
        coverage_map = torch.zeros((H, W)).to(patch_pred.device)
        for (x, y), pred in zip(xy_inds, patch_pred):
            pixel_pred[y : y + self.patch_size, x : x + self.patch_size] += pred
            coverage_map[y : y + self.patch_size, x : x + self.patch_size] += 1
        # perform gaussian smoothing
        pixel_pred = gaussian_filter_2d(pixel_pred, sigma=32)
        coverage_map = gaussian_filter_2d(coverage_map, sigma=32)
        pixel_pred /= coverage_map + 1e-8
        pixel_pred /= pixel_pred.max() + 1e-8
        if pixel_pred.sum() > pixel_pred.numel() * 0.5:
            pixel_pred = 1 - pixel_pred
        pixel_pred = (pixel_pred > self.loc_threshold).float()
        return pixel_pred
