import warnings
import torch
import torch.nn.functional as F
from numba import jit
from .convert_utils import gaussian_filter_2d
from .plwrapper import FsmPLWrapper


def batch_fn(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


class FSG(FsmPLWrapper):
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

    with torch.no_grad():
        img_preds, loc_preds = model(image[None, ...].to(model.device))

    plt.imshow(loc_preds.cpu()[0])
    plt.colorbar()
    plt.show()
    ```
    """

    def __init__(
        self,
        stride_ratio=0.5,
        fast_sim_mode=False,
        loc_threshold=0.37,
        is_high_sim=True,
        need_input_255=False,
        **kwargs,
    ):
        """
        Initializes the Forensic Similarity Graph (FSG) algorithm with specific parameters.

        Parameters:
        - stride_ratio (float): The ratio of the stride to the patch size, determining the overlap between patches. The lower the value, the higher the overlap.
        - fast_sim_mode (bool): If True, the algorithm uses a faster method to compute similarity scores, potentially at the cost of accuracy.
        - loc_threshold (float): The threshold for determining the location of interest in the similarity graph. Values above this threshold are considered significant.
        - is_high_sim (bool): If True, higher similarity scores indicate higher similarity. If False, lower scores indicate higher similarity.
        - need_input_255 (bool): If True, input images are expected to be scaled to [0, 255]. If False, images are expected to be in [0, 1].
        - **kwargs: Additional keyword arguments passed to the superclass initializer.

        This class is designed to create a graph-based representation of forensic similarity between different patches of an image, allowing for the detection of manipulated regions.
        """
        super().__init__(**kwargs)
        self.patch_size = self.model.fe.patch_size
        self.stride = int(self.patch_size * stride_ratio)
        self.fast_sim_mode = fast_sim_mode
        self.loc_threshold = loc_threshold
        self.is_high_sim = is_high_sim
        self.need_input_255 = need_input_255
        warnings.filterwarnings("ignore")

    def get_batched_patches(self, x: torch.Tensor):
        B, C, H, W = x.shape
        if self.need_input_255:
            x = x * 255
        # split images into batches of patches: B x C x H x W -> B x (NumPatchHeight x NumPatchWidth) x C x PatchSize x PatchSize
        batched_patches = (
            x.unfold(2, self.patch_size, self.stride)
            .unfold(3, self.patch_size, self.stride)
            .permute(0, 2, 3, 1, 4, 5)
        )
        batched_patches = batched_patches.contiguous().view(B, -1, C, self.patch_size, self.patch_size)
        return batched_patches

    @jit(forceobj=True)
    def get_features(self, image_patches: torch.Tensor):
        patches_features = []
        for batch in list(batch_fn(image_patches, 256)):
            batch = batch.float().to(self.device)
            feats = self.model.fe(batch).detach().cpu()
            patches_features.append(feats)
        patches_features = torch.vstack(patches_features)
        return patches_features

    @jit(forceobj=True)
    def get_sim_scores(self, patch_pairs):
        patches_sim_scores = []
        for batch in list(batch_fn(patch_pairs, 4096)):
            batch = batch.permute(1, 0, 2).float().to(self.device)
            scores = self.model.comparenet(*batch).detach().cpu()
            scores = torch.nn.functional.softmax(scores, dim=1)
            patches_sim_scores.append(scores)
        patches_sim_scores = torch.vstack(patches_sim_scores)
        return patches_sim_scores

    def forward_single(self, patches: torch.Tensor):
        P, C, H, W = patches.shape
        features = self.get_features(patches)
        sim_mat = torch.zeros(P, P)
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
        return img_pred.detach().to(self.device), patch_pred.detach().to(self.device)

    @jit(forceobj=True)
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        # get the (x, y) coordinates of the top left of each patch in the image
        x_inds = torch.arange(W).unfold(0, self.patch_size, self.stride)[:, 0]
        y_inds = torch.arange(H).unfold(0, self.patch_size, self.stride)[:, 0]
        xy_inds = torch.tensor([(ii, jj) for jj in y_inds for ii in x_inds]).to(self.device)

        batched_patches = self.get_batched_patches(x)
        img_preds = []
        loc_preds = []
        for patches in batched_patches:
            img_pred, patch_pred = self.forward_single(patches)
            img_preds.append(img_pred)
            loc_pred = self.patch_to_pixel_pred(patch_pred, xy_inds)
            loc_pred = F.interpolate(loc_pred[None, None, ...], size=(H, W), mode="nearest").squeeze()
            loc_preds.append(loc_pred)
        img_preds = torch.tensor(img_preds)
        loc_preds = torch.stack(loc_preds)
        return img_preds, loc_preds

    def patch_to_pixel_pred(self, patch_pred, xy_inds):
        W, H = torch.max(xy_inds, dim=0).values + self.patch_size
        pixel_pred = torch.zeros((H, W)).to(self.device)
        coverage_map = torch.zeros((H, W)).to(self.device)
        for (x, y), pred in zip(xy_inds, patch_pred):
            pixel_pred[y : y + self.patch_size, x : x + self.patch_size] += pred
            coverage_map[y : y + self.patch_size, x : x + self.patch_size] += 1
        # perform gaussian smoothing
        pixel_pred = gaussian_filter_2d(pixel_pred, sigma=32)
        coverage_map = gaussian_filter_2d(coverage_map, sigma=32)
        pixel_pred /= coverage_map + 1e-8
        pixel_pred /= pixel_pred.max() + 1e-8
        pixel_pred = (pixel_pred > self.loc_threshold).float()
        if pixel_pred.sum() > pixel_pred.numel() * 0.5:
            pixel_pred = 1 - pixel_pred
        return pixel_pred
