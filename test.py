import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# sys.path.append('./scripts')
from scripts.model import UNetEfficientNet, UNetEfficientNet_Skip_ELA, UNetEfficientNet_Skip
from data import image_transform

#=====================================
# Load Model Function
#=====================================
def load_model(model_path, model_name='unet_effnet', device=None):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path (str): Path to the .pth checkpoint
        model_name (str): One of 'unet_effnet', 'unet_effnet_ela', 'unet_effnet_cbam'
        device (torch.device): Device to load the model on (optional)
    
    Returns:
        model (torch.nn.Module): Loaded model in eval mode
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'unet_effnet':
        model_class = UNetEfficientNet
    elif model_name == 'unet_effnet_ela':
        model_class = UNetEfficientNet_Skip_ELA #ela 
    elif model_name == 'unet_effnet_skip':
        model_class = UNetEfficientNet_Skip  # or UNetEfficientNet_CBAM
    
    model = model_class(num_classes=1, encoder_name='efficientnet-b5', pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


def predict_single_image(model, image_path, device=None, transform=None):
    """
    Predict a mask for a single image.
    
    Args:
        model (torch.nn.Module): Trained segmentation model
        image_path (str): Path to the input image
        device (torch.device): Device to run inference on
        transform (callable): Optional image transformation (if None, uses default)
    
    Returns:
        mask_pred (torch.Tensor): Predicted mask as tensor (H x W)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if transform is None:
        transform = image_transform  
    
    image = Image.open(image_path).convert('RGB')

    image_tensor = transform(image).unsqueeze(0)  # add batch dimension
    image_tensor = image_tensor.to(device)
    
    # forward pass
    with torch.no_grad():
        pred = model(image_tensor)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]  # some models return tuple
        mask_pred = torch.sigmoid(pred).squeeze(0).cpu()  
  
        if mask_pred.shape != image.size[::-1]:  # PIL size is (W, H)
            mask_pred = transforms.functional.resize(mask_pred.unsqueeze(0), image.size[::-1], interpolation=transforms.InterpolationMode.BILINEAR).squeeze(0)
        #
    return image, mask_pred

#=====================================
# Get predictions
#=====================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_mask_probability(model_path, model_name, image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, model_name=model_name, device=device)
    image, mask_pred = predict_single_image(model, image_path, device=device)
    mask_pred = mask_pred.numpy()
    mask_pred = np.transpose(mask_pred, (1, 2, 0))
    mask_prob = mask_pred.squeeze()
    # mask_pred = mask_pred > 0.9
    
    return image, mask_pred, mask_prob, model

baseline_model_path = 'models/weak/baseline/best_model_fold_2.pth'
skip_model_path = 'models/weak/baseline_skip/best_model_fold_2.pth'
skip_attention_model_path = 'models/weak/baseline_skip_ela/best_model_fold_2.pth'

#call the model
img, mask_pred, mask_prob, model = get_mask_probability(baseline_model_path, 'unet_effnet', 'examples/0af4a2ad0b.png')
img2, mask_pred2, mask_prob2, model2 = get_mask_probability(skip_model_path, 'unet_effnet_skip', 'examples/0af4a2ad0b.png')
img3, mask_pred3, mask_prob3, model3 = get_mask_probability(skip_attention_model_path, 'unet_effnet_ela', 'examples/0af4a2ad0b.png')

#=====================================
#Visualization
#=====================================
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(15, 3))

ax1.imshow(img3, cmap='gray')
ax1.set_title("Original Image")
ax1.axis('off')

# Salt probability maps thresholds
thresholds = [0.3, 0.5, 0.7, 0.9]
titles = ["threshold=0.3", "threshold=0.5", "threshold=0.7", "threshold=0.9"]

axes = [ax2, ax3, ax4, ax5]
vmin, vmax = 0, 1

for ax, th, title in zip(axes, thresholds, titles):
    ax.imshow(img3, cmap='gray')  # base seismic image
    im = ax.imshow(mask_pred3 > th, cmap='jet', alpha=0.5, vmin=vmin, vmax=vmax)  
    ax.set_title(title)
    ax.axis('off')

divider = make_axes_locatable(ax5)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax, label='Salt Probabilities')

plt.subplots_adjust(wspace=0.05)
plt.tight_layout()
plt.show()
