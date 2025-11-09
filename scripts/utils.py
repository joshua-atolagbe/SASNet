import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Set seeds for reproducibility
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def compute_coverage(mask_path):
    """Calculate salt coverage percentage for a mask"""
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = (mask > 0).astype(np.float32)
    coverage = (mask.sum() / mask.size) * 100
    return coverage

def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i:
            return i
    return 10

# Plotting functions
def plot_training_metrics(fold_results):
    """Plot training metrics for all folds"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot losses
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    
    # Plot IoU
    axes[0, 1].set_title('Validation IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    
    # Plot Pixel Accuracy
    axes[1, 0].set_title('Validation Pixel Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Pixel Accuracy')
    
    # Plot FWIoU
    axes[1, 1].set_title('Validation Frequency Weighted IoU')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('FWIoU')
    
    for fold, results in enumerate(fold_results):
        epochs = range(1, len(results['train_losses']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, results['train_losses'], label=f'Fold {fold+1} Train', alpha=0.7)
        axes[0, 0].plot(epochs, results['val_losses'], label=f'Fold {fold+1} Val', alpha=0.7)
        
        # IoU plot
        axes[0, 1].plot(epochs, results['val_ious'], label=f'Fold {fold+1}', alpha=0.7)
        
        # Pixel Accuracy plot
        axes[1, 0].plot(epochs, results['val_pixel_accs'], label=f'Fold {fold+1}', alpha=0.7)
        
        # FWIoU plot
        axes[1, 1].plot(epochs, results['val_fwious'], label=f'Fold {fold+1}', alpha=0.7)
    
    for ax in axes.flat:
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_predictions(model, test_loader, device, num_samples=6):
    """Plot test predictions"""
    model.eval()
    
    fig, axes = plt.subplots(3, num_samples, figsize=(18, 9))
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            images = images.to(device)
            outputs = model(images)
            
            # Get first sample from batch
            image = images[0].cpu()
            mask = masks[0].cpu()
            pred = torch.sigmoid(outputs[0]).cpu()
            
            # Denormalize image for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
            image = torch.clamp(image, 0, 1)
            
            # Plot original image
            axes[0, i].imshow(image.permute(1, 2, 0))
            axes[0, i].set_title('Original Image')
            axes[0, i].axis('off')
            
            # Plot ground truth mask
            axes[1, i].imshow(mask.squeeze(), cmap='gray')
            axes[1, i].set_title('Ground Truth')
            axes[1, i].axis('off')
            
            # Plot prediction
            axes[2, i].imshow(pred.squeeze(), cmap='gray')
            axes[2, i].set_title('Prediction')
            axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()

