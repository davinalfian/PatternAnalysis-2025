import torch

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