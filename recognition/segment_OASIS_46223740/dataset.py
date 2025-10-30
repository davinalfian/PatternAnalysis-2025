import os
import torch
import numpy as np
import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import Dataset


class OASISDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_size=(256, 256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_size = target_size

        self.images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.png')])
        self.masks = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith('.png')])

        assert len(self.images) ==len(self.masks), "Image/mask count mismtach"

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        mask = np.array(mask) // 85

        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.int64)

        # Normalize image: scale to [0, 1] then standardize
        img = img / 255.0
        img = (img - img.mean()) / (img.std() + 1e-8)


        # to tensors: (C, H, W) for image - (H, W) for mask
        img_t = torch.from_numpy(img).unsqueeze(0)          # shape (1, H, H)
        mask_t = torch.from_numpy(mask).long()              # shape (H, W), dtype long


        # Optional transform
        if self.transform:
            img_t, mask_t = self.transform(img_t, mask_t)
        
        return img_t.float(), mask_t.long()
    

# train_dir = "OASIS/keras_png_slices_train"

# for filename in os.listdir(train_dir):
#     if filename.endswith('.png'):
#         img_path = os.path.join(train_dir, filename)

#         mask = np.array(Image.open(img_path))
#         print(filename, np.unique(mask))
        
#         img = Image.open(img_path)
#         print(filename, img.mode)
#         break

# print("\n ============= \n")

# seg_dir = "OASIS/keras_png_slices_seg_train"

# for filename in os.listdir(seg_dir):
#     if filename.endswith('.png'):
#         img_path = os.path.join(seg_dir, filename)

#         mask = np.array(Image.open(img_path))
#         print(filename, np.unique(mask))

#         img = Image.open(img_path)
#         print(filename, img.mode)

#         break

# print("\n ============= \n")

# train_files = sorted(os.listdir(train_dir))
# seg_files = sorted(os.listdir(seg_dir))

# for t, s in zip(train_files[:5], seg_files[:5]):
#     print(t, '<->', s)


## Classes are [0 85 170 255]

# if torch.cuda.is_available():
#     print("GPU is available!")
#     print(f"Number of GPUs: {torch.cuda.device_count()}")
#     print(f"GPU Name: {torch.cuda.get_device_name(0)}") # For the first GPU
# else:
#     print("GPU is not available. PyTorch will use the CPU.")