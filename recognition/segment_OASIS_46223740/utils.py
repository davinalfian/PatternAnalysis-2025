import torch
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