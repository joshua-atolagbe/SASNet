from PIL import Image
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset

IMAGE_SIZE = (128, 128)

image_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),  
    transforms.RandomVerticalFlip()
])


class SaltDataset(Dataset):
    def __init__(self, image_dir, mask_dir, df,
                  image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.df = df
        self.image_transform = image_transform
        self.mask_transform = mask_transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['id']
        
        img_path = os.path.join(self.image_dir, f"{img_name}.png")
        image = Image.open(img_path).convert('RGB')
        
        mask_path = os.path.join(self.mask_dir, f"{img_name}.png")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
        else:
            mask = Image.new('L', image.size, 0)
        
        if self.image_transform:
            image = self.image_transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return image, mask