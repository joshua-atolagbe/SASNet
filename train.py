import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader

from scripts.model import UNetEfficientNet, UNetEfficientNet_Skip, UNetEfficientNet_Skip_ELA
from scripts.engine import train_model
from scripts.metrics import DiceBCELoss, WeakFocalLoss
from scripts.utils import seed_everything
from scripts.data import SaltDataset, image_transform, mask_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net with EfficientNet for Salt Segmentation")
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--image_size', type=tuple, default=(128, 128))
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--train_csv', type=str, default='../stratify_weak.csv')
    parser.add_argument('--model_save_dir', type=str, help='Directory to save models')
    parser.add_argument('--learning_type', type=str, choices=['weak', 'strong'], default='weak')
    parser.add_argument('--model', type=str, choices=['unet_effnet_skip_ela', 'unet_effnet', 'unet_effnet_skip'], default='unet_effnet')
    # parser.add_argument('--display_results', action='store_true')

    return parser.parse_args()

def main(args):
    seed_everything(42)
    train_df = pd.read_csv(args.train_csv)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=2025)
    
    fold_results = []
    
    if args.model == 'unet_effnet':
        model_class = UNetEfficientNet
    elif args.model == 'unet_effnet_skip_ela':
        model_class = UNetEfficientNet_Skip_ELA
    elif args.model == 'unet_effnet_skip':
        model_class = UNetEfficientNet_Skip
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['coverage_class'])):
        print(f"\n{'='*60}\nFOLD {fold + 1}\n{'='*60}")
        
        train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)
        
        train_dataset = SaltDataset('data/images', 'data/new_masks', train_fold_df, 
                                   image_transform=image_transform, mask_transform=mask_transform)
        val_dataset = SaltDataset('data/images', 'data/new_masks', val_fold_df,
                                 image_transform=image_transform, mask_transform=mask_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        model = model_class(num_classes=1, encoder_name='efficientnet-b5', pretrained=True)
        criterion = DiceBCELoss() if args.learning_type == 'strong' else WeakFocalLoss(beta=1)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        
        results = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, args.num_epochs, device)
        fold_results.append(results)
        
        os.makedirs('models/weak/'+args.model_save_dir, exist_ok=True)
        results_df = pd.DataFrame({
            'epoch': list(range(1, args.num_epochs + 1)),
            'train_loss': results['train_losses'],
            'train_iou': results['train_ious'],
            # 'train_wba': results['train_wbas'],          # updated
            'train_fw_iou': results['train_fwious'],
            'val_loss': results['val_losses'],
            'val_iou': results['val_ious'],
            # 'val_wba': results['val_wbas'],             # updated
            'val_fw_iou': results['val_fwious']
        })
        results_df.to_csv(f'models/weak/{args.model_save_dir}/fold_{fold+1}_results.csv', index=False)

        torch.save(results['best_model_state'], f'models/weak/{args.model_save_dir}/best_model_fold_{fold+1}.pth')
        
        print(f"Fold {fold+1} Best Val Loss: {results['best_val_loss']:.4f}")

    
    return fold_results, train_fold_df, val_fold_df

if __name__ == "__main__":
    args = parse_args()
    fold_results, trainfold, valfold = main(args)
