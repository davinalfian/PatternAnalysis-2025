from modules import ImprovedUNet
from dataset import OASISDataset
from utils import dice_coefficient_per_class

import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")

model = ImprovedUNet(n_channels=1, n_classes=4)
model = model.to(device)

# dir = "/home/groups/comp3710/"
test_dataset = OASISDataset("OASIS/keras_png_slices_test", "OASIS/keras_png_slices_seg_test")
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)

checkpoint = torch.load("./oasis_checkpoint.pth", weights_only=False, map_location=device)
best_epoch = checkpoint['epoch']

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


print("\nTest start...")
start_time = time.time()

dice_scores = []
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        dice_score_per_class = dice_coefficient_per_class(outputs, masks)
        
        dice_scores.append(dice_score_per_class)

    test_dice = np.nanmean(dice_scores)
    test_dice_per_class = np.nanmean(dice_scores, axis=0)


print(f"Best Epoch {best_epoch}")
print(f"  Test Dice Score: {test_dice:.4f} \n")

for i, score in enumerate(test_dice_per_class):
    print(f"Class {i} - Dice score: {score}")

end_time = time.time()
print(f"\nTotal time: {(end_time - start_time)/60:.2f}")