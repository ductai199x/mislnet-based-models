import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from pillow_heif import register_heif_opener
from torchvision.transforms.functional import resize, pil_to_tensor, crop
from kornia.augmentation import (
    AugmentationSequential,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomBrightness,
    RandomContrast,
    RandomSaturation,
    RandomSharpness,
    RandomGaussianBlur,
    RandomBoxBlur,
    RandomJPEG,
    RandomGaussianNoise,
)


register_heif_opener()
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageConsistencyDataset(Dataset):
    def __init__(
        self,
        root_dir,
        image_paths,
        patch_size,
        patch_per_dir=5,
        is_augment_input=False,
    ):
        self.root_dir = root_dir
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.patch_per_dir = patch_per_dir
        self.is_augment_input = is_augment_input
        self.augmentations = AugmentationSequential(
            RandomHorizontalFlip(p=1.0, keepdim=True),
            RandomVerticalFlip(p=1.0, keepdim=True),
            RandomBrightness(brightness=(0.7, 1.3), p=1.0, keepdim=True),
            RandomContrast(contrast=(0.7, 1.3), p=1.0, keepdim=True),
            RandomSaturation(saturation=(0.7, 1.3), p=1.0, keepdim=True),
            RandomSharpness(sharpness=0.5, p=1.0, keepdim=True),
            RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.5, 1.5), p=1.0, keepdim=True),
            RandomBoxBlur(kernel_size=(5, 5), p=1.0, keepdim=True),
            RandomJPEG(jpeg_quality=(50, 100), p=1.0, keepdim=True),
            RandomGaussianNoise(mean=0, std=1 / 15, p=1.0, keepdim=True),
            random_apply=1,
            keepdim=True,
        )

    def __len__(self):
        return len(self.image_paths)

    def sample_center_patch_idxs(self, image):
        C, H, W = image.shape
        x_idxs = torch.randint(self.patch_size, W - self.patch_size * 2, (self.patch_per_dir,))
        y_idxs = torch.randint(self.patch_size, H - self.patch_size * 2, (self.patch_per_dir,))
        idxs = torch.cartesian_prod(y_idxs, x_idxs)
        return idxs

    def get_neighbor_patch_idxs(self, indices):
        neighbor_idxs_of_center_patch = []
        for y, x in indices:
            tl_patch = (y - self.patch_size, x - self.patch_size)
            tr_patch = (y - self.patch_size, x + self.patch_size)
            bl_patch = (y + self.patch_size, x - self.patch_size)
            br_patch = (y + self.patch_size, x + self.patch_size)
            t_patch = (y - self.patch_size, x)
            b_patch = (y + self.patch_size, x)
            l_patch = (y, x - self.patch_size)
            r_patch = (y, x + self.patch_size)
            neighbor_idxs_of_center_patch.append(
                [tl_patch, tr_patch, bl_patch, br_patch, t_patch, b_patch, l_patch, r_patch]
            )
        return torch.LongTensor(neighbor_idxs_of_center_patch)

    def __getitem__(self, idx):
        image_path = f"{self.root_dir}/{self.image_paths[idx]}"
        image = pil_to_tensor(Image.open(image_path).convert("RGB")).float().div(255)
        C, H, W = image.shape
        # crop_H, crop_W = (H // 16) * 16, (W // 16) * 16
        # image = crop(image, 0, 0, crop_H, crop_W)
        crop_H, crop_W = min((H // 16) * 16, 2160), min((W // 16) * 16, 2160)
        top_left_x = 0 if crop_W == W else torch.randint(W - crop_W, (1,)).item()
        top_left_y = 0 if crop_H == H else torch.randint(H - crop_H, (1,)).item()
        image = crop(image, top_left_y, top_left_x, crop_H, crop_W)
        if H < self.patch_size * 4 or W < self.patch_size * 4:
            new_H = max(H, self.patch_size * 4)
            new_W = max(W, self.patch_size * 4)
            image = resize(image, (new_H, new_W))
            H, W = new_H, new_W
        center_patch_idxs = self.sample_center_patch_idxs(image)
        neighbor_patch_idxs = self.get_neighbor_patch_idxs(center_patch_idxs)
        center_patches = []
        for center_patch in neighbor_patch_idxs:
            neighbor_patches = []
            for y, x in center_patch:
                patch = image[:, y : y + self.patch_size, x : x + self.patch_size]
                neighbor_patches.append(patch)
            neighbor_patches = torch.stack(neighbor_patches)
            center_patches.append(neighbor_patches)
        center_patches = torch.stack(center_patches)

        if self.is_augment_input:
            # crop_H, crop_W = min((H // 16) * 16, 1024), min((W // 16) * 16, 1024)
            # top_left_x = 0 if crop_W == W else torch.randint(W - crop_W, (1,)).item()
            # top_left_y = 0 if crop_H == H else torch.randint(H - crop_H, (1,)).item()
            # aug_image = crop(image, top_left_y, top_left_x, crop_H, crop_W)
            aug_image = self.augmentations(image.unsqueeze(0)).squeeze(0)
            aug_image = (torch.clamp(aug_image, 0, 1) * 255).int().float().div(255)
            center_patch_idxs = self.sample_center_patch_idxs(aug_image)
            neighbor_patch_idxs = self.get_neighbor_patch_idxs(center_patch_idxs)
            aug_center_patches = []
            for center_patch in neighbor_patch_idxs:
                neighbor_patches = []
                for y, x in center_patch:
                    patch = aug_image[:, y : y + self.patch_size, x : x + self.patch_size]
                    neighbor_patches.append(patch)
                neighbor_patches = torch.stack(neighbor_patches)
                aug_center_patches.append(neighbor_patches)
            aug_center_patches = torch.stack(aug_center_patches)
            return center_patches, aug_center_patches
        else:
            return center_patches
