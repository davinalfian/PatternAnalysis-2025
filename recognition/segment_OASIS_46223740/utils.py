import torch
import random
import matplotlib.pyplot as plt

def dice_coefficient(pred, target, num_classes=4, epsilon=1e-6):
    dice = 0.0
    pred = torch.argmax(pred, dim=1).to(target.device)

    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        dice += (2. * intersection + epsilon) / (union + epsilon)
    
    return dice / num_classes

def dice_coefficient_per_class(pred, target, num_classes=4, epsilon=1e-6):
    dice_per_class = []
    pred = torch.argmax(pred, dim=1).to(target.device)

    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2. * intersection + epsilon) / (union + epsilon)

        dice_per_class.append(dice.item())
    
    return dice_per_class

def training_plots(performance_tracking):
    num_epoch = len(performance_tracking['dice_score'])
    performance_tracking['dice_score'] = [score.cpu().numpy() for score in performance_tracking['dice_score']]

    plt.figure(figsize=(10, 10))
    plt.plot(range(num_epoch), performance_tracking['train_loss'], label='training_loss')
    plt.plot(range(num_epoch), performance_tracking['val_loss'], label='validation_loss')
    plt.legend(loc='upper right')
    plt.title('Training Loss and Validation Loss')
    plt.savefig('unet_output/unet-model-loss.png')

    plt.figure(figsize=(10, 10))
    plt.plot(range(num_epoch), performance_tracking['dice_score'], label='training_loss')
    plt.legend(loc='upper right')
    plt.title('Dice scores')
    plt.savefig('unet_output/unet-dice-scores.png')

def brain_visualization(model, dataset, device, num_samples=3, indices=None):
    if indices is None:
        random.seed(26)
        indices = random.sample(range(len(dataset)), num_samples)
    
    _, axes = plt.subplots(num_samples, 3, figsize=(12, 12))
    columns = ["Image", "Ground Truth", "Prediction"]

    for col, title in enumerate(columns):
        axes[0, col].set_title(title, fontsize=14)
    
    with torch.no_grad():
        for row, idx in enumerate(indices):
            image, mask = dataset[idx]

            input_tensor = image.unsqueeze(0).to(device)
            output = model(input_tensor)
            pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

            image_np = image.squeeze(0).cpu().numpy()
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

            mask_np = mask.cpu().numpy()

            mask_np = mask_np * 85
            pred_mask = pred_mask * 85

            axes[row, 0].imshow(image_np, cmap='gray')
            axes[row, 0].axis("off")

            axes[row, 1].imshow(mask_np, cmap='gray', interpolation="nearest")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(pred_mask, cmap='gray', interpolation="nearest")
            axes[row, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig('unet_output/unet-brains.png')