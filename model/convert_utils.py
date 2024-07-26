import math
import torch
from torch.nn import functional as F
from torch.distributions import Normal
from .utils import split_image_into_patches


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


def convert_sim_mat_to_detection(sim_mat: torch.Tensor):
    sim_mat = torch.clamp(sim_mat, 0, 1)
    # sim_mat = 0.5 * (sim_mat + sim_mat.T)
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


def patch_pred_to_pixel_pred(patch_pred: torch.Tensor, patch_size: int, stride: int, image_size: tuple):
    H, W = image_size
    x_inds = torch.arange(W).unfold(0, patch_size, stride)[:, 0]
    y_inds = torch.arange(H).unfold(0, patch_size, stride)[:, 0]
    xy_inds = torch.tensor([(ii, jj) for jj in y_inds for ii in x_inds])
    pixel_pred = torch.zeros((H, W))
    coverage_map = torch.zeros((H, W))
    for i, (x, y) in enumerate(xy_inds):
        pixel_pred[y : y + patch_size, x : x + patch_size] += patch_pred[i]
        coverage_map[y : y + patch_size, x : x + patch_size] += 1
    pixel_pred = gaussian_filter_2d(pixel_pred, sigma=8)
    coverage_map = gaussian_filter_2d(coverage_map, sigma=8)
    pixel_pred = pixel_pred / (coverage_map + 1e-8)
    return pixel_pred


def convert_gt_mask_to_sim_mat(gt_mask: torch.Tensor, patch_size: int, stride: int):
    gt_mask_patches, num_patch_h, num_patch_w = split_image_into_patches(gt_mask, patch_size, stride)
    gt_mask_scores = (gt_mask_patches.squeeze(1).float().mean(dim=(1, 2)) > 0.5).int()
    a = torch.randn(1, 3)
    b = torch.randn(1, 3)
    c = torch.cross(a, b, dim=1)
    gt_mask_vecs = torch.zeros(gt_mask_scores.shape[0], 3)
    gt_mask_vecs[gt_mask_scores == 1] = a
    gt_mask_vecs[gt_mask_scores == 0] = c
    gt_sim_mat = F.normalize(gt_mask_vecs, p=2, dim=1) @ F.normalize(gt_mask_vecs, p=2, dim=1).T
    return gt_sim_mat
