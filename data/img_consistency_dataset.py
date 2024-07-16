import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import resize
from torch.utils.data import Dataset


class ImageConsistencyDataset(Dataset):
    def __init__(
        self,
        root_dir,
        image_paths,
        patch_size,
        patch_per_dir=5,
    ):
        self.root_dir = root_dir
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.patch_per_dir = patch_per_dir

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
        image = read_image(image_path, ImageReadMode.RGB).float().div(255)
        C, H, W = image.shape
        if H < self.patch_size * 4 or W < self.patch_size * 4:
            new_H = max(H, self.patch_size * 4)
            new_W = max(W, self.patch_size * 4)
            image = resize(image, (new_H, new_W))
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
        return torch.stack(center_patches)
