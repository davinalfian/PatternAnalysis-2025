from modules import ImprovedUNet
from dataset import OASISDataset
from utils import dice_coefficient

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")

# "/home/groups/comp3710/OASIS"
dir = "/home/groups/comp3710/"
train_dataset = OASISDataset(dir + "OASIS/keras_png_slices_train", dir + "OASIS/keras_png_slices_seg_train")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

val_dataset = OASISDataset(dir + "OASIS/keras_png_slices_validate", dir + "OASIS/keras_png_slices_seg_validate")
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)

# Classes color index are [0 85 170 255]
model = ImprovedUNet(n_channels=1, n_classes=4)
model = model.to(device)

lr = 1e-4
epochs = 30
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

performance_tracking = { 'train_loss': [], 'val_loss': [], 'dice_score': [] }
best_dice = 0.0

start_time = time.time()
for epoch in range(epochs):

    ## Training ##
    start = time.time()
    train_loss = 0.0

    model.train()
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)


    ## Validation ##
    val_loss = 0.0
    dice_score = 0.0
    
    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item()
            dice_score += dice_coefficient(outputs, masks)

        avg_val_loss = val_loss / len(val_loader)
        avg_dice = dice_score / len(val_loader)

    end = time.time()


    print(f"Epoch {epoch+1}:")
    print(f"   Time Taken:  {(end - start)/60:.2f}")
    print(f"   Train Loss:  {avg_train_loss}")
    print(f"   Val Loss:    {avg_val_loss}")
    print(f"   Dice Loss:   {avg_dice}")

    performance_tracking['train_loss'].append(avg_train_loss)
    performance_tracking['val_loss'].append(avg_val_loss)
    performance_tracking['dice_score'].append(avg_dice)

    if avg_dice > best_dice:
        best_dice = avg_dice
        checkpoint = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
            'performance_tracking': performance_tracking
        }

end_time = time.time()
torch.save(checkpoint, 'oasis_checkpoint.pth')

print(f"\nTotal time: {(end_time - start_time)/60:.2f}")


# plt.figure(figsize=(10, 10))
# plt.plot(range(epochs), performance_tracking['train_loss'], label='training_loss')
# plt.plot(range(epochs), performance_tracking['val_loss'], label='validation_loss')
# plt.legend(loc='upper right')
# plt.title('Training Loss and Validation Loss')
# plt.savefig('fcn-output/fcn-model-loss.png')