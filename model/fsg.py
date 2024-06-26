import math
import warnings
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from numba import jit

from .fsm_plwrapper import FsmPLWrapper
from typing import *


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


class FSG(FsmPLWrapper):
    def __init__(self, stride_ratio=0.5, fast_sim_mode=False, loc_threshold=0.37, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = self.model.fe.patch_size
        self.stride = int(self.patch_size * stride_ratio)
        self.fast_sim_mode = fast_sim_mode
        self.loc_threshold = loc_threshold
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
