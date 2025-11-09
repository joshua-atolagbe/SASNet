import torch
from tqdm import tqdm
from metrics import iou_score, frequency_weighted_iou


# Training function with metrics tracking
def train_model(model, train_loader, val_loader, criterion, optimizer,
                 scheduler, num_epochs=25, device='cuda'):
    model.to(device)
    
    # Metrics tracking
    train_losses = []
    train_ious = []
    # train_wbas = []          # weighted balanced accuracy
    train_fwious = []
    val_losses = []
    val_ious = []
    # val_wbas = []            # weighted balanced accuracy
    val_fwious = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        # running_wba = 0.0
        running_fwiou = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{num_epochs}')
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device).float()
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_iou += iou_score(outputs, masks)
            # running_wba += weighted_balanced_accuracy(outputs, masks)
            running_fwiou += frequency_weighted_iou(outputs, masks)
            
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'iou': f"{running_iou / (train_pbar.n + 1):.4f}",
                # 'wba': f"{running_wba / (train_pbar.n + 1):.4f}"
            })

        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        # val_wba = 0.0
        val_fwiou = 0.0
        
        val_pbar = tqdm(val_loader, desc=f'Val Epoch {epoch+1}/{num_epochs}')
        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device).float()
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_iou += iou_score(outputs, masks)
                # val_wba += weighted_balanced_accuracy(outputs, masks)
                val_fwiou += frequency_weighted_iou(outputs, masks)
                
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'iou': f"{val_iou / (val_pbar.n + 1):.4f}",
                    'wba': f"{val_wba / (val_pbar.n + 1):.4f}"
                })

        
        # Calculate epoch metrics
        train_loss = running_loss / len(train_loader)
        train_iou = running_iou / len(train_loader)
        # train_wba = running_wba / len(train_loader)
        train_fwiou = running_fwiou / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_iou = val_iou / len(val_loader)
        val_wba = val_wba / len(val_loader)
        val_fwiou = val_fwiou / len(val_loader)
        
        # Store metrics
        train_losses.append(train_loss)
        train_ious.append(train_iou)
        # train_wbas.append(train_wba)
        train_fwious.append(train_fwiou)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        # val_wbas.append(val_wba)
        val_fwious.append(val_fwiou)
        
        scheduler.step(val_loss)
        
        # Save best model based on lowest validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f'New best model saved with Val Loss: {best_val_loss:.4f}')
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        # print(f'Train IoU: {train_iou:.4f}, Train WBA: {train_wba:.4f}, Train FWIoU: {train_fwiou:.4f}')
        print(f'Val IoU: {val_iou:.4f}, Val FWIoU: {val_fwiou:.4f}')
        print('-' * 60)
    
    return {
        'train_losses': train_losses,
        'train_ious': train_ious,
        # 'train_wbas': train_wbas,
        'train_fwious': train_fwious,
        'val_losses': val_losses,
        'val_ious': val_ious,
        # 'val_wbas': val_wbas,
        'val_fwious': val_fwious,
        'best_model_state': best_model_state,
        'best_val_loss': best_val_loss
    }