import torch


def split_image_into_patches(x: torch.Tensor, patch_size: int, stride: int):
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    B, C, H, W = x.shape
    # split images into batches of patches: B x C x H x W -> B x (NumPatchHeight x NumPatchWidth) x C x PatchSize x PatchSize
    batched_patches = (
        x.unfold(2, patch_size, stride)
        .unfold(3, patch_size, stride)
        .permute(0, 2, 3, 1, 4, 5)
    )
    _, num_patch_h, num_patch_w, _, _, _ = batched_patches.shape
    batched_patches = batched_patches.contiguous().view(B, -1, C, patch_size, patch_size)
    if B == 1:
        batched_patches = batched_patches.squeeze(0)
    return batched_patches, num_patch_h, num_patch_w