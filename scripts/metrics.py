import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Loss function - Dice Loss + BCE
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
    
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE


# for handling weak/noisy labels
class WeakFocalLoss(nn.Module):
    """
    Weak Focal Loss adapted for binary segmentation with confidence values.
    Based on the Alaudah et al.:
    WFL(q(x), p(x)) = -∑(1-pn(x))^γ * qn(x)^α * log(pn(x))
    
    Args:
        alpha: Controls the contribution of confidence values (default: 1.0)
        gamma: Focusing parameter for hard examples (default: 2.0)
        beta: Similarity scaling parameter (default: 1.0)
        reduction: Specifies reduction to apply to output ('mean', 'sum', 'none')
    """
    def __init__(self, alpha=1.0, gamma=2.0, beta=1.0, reduction='mean'):
        super(WeakFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, inputs, targets, confidence=None, similarity=None):

        probs = torch.sigmoid(inputs)
        
        if confidence is None:
            confidence = targets.clone()
        confidence = confidence.to(inputs.device)
        
        # Positive class loss
        pos_loss = -((1 - probs) ** self.gamma) * (confidence ** self.alpha) * torch.log(probs + 1e-8)
        
        # Negative class loss
        neg_confidence = 1.0 - confidence
        neg_loss = -(probs ** self.gamma) * (neg_confidence ** self.alpha) * torch.log(1 - probs + 1e-8)
        
        focal_loss = targets * pos_loss + (1 - targets) * neg_loss
        
        # apply similarity scaling of 1
        if similarity is None:
            similarity = torch.ones(inputs.size(0), device=inputs.device)
        similarity = similarity.view(-1, 1, 1, 1).expand_as(focal_loss)
        focal_loss = focal_loss * (similarity ** self.beta)
        
        # reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Metrics
def iou_score(output, target):
    smooth = 1e-5
    
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    
    return (intersection + smooth) / (union + smooth)

def frequency_weighted_iou(output, target, num_classes=2):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    output_ = (output > 0.5).astype(int)
    target_ = target.astype(int)
    
    hist = np.bincount(num_classes * target_.flatten() + output_.flatten(),
                        minlength=num_classes**2).reshape(num_classes, num_classes)
    ious = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    freq = hist.sum(axis=1) / hist.sum()
    
    return np.nansum(freq * ious)