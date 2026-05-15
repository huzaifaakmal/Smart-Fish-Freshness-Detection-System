import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import yaml
import os
import sys
from tqdm import tqdm
import time

# Add root directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.fish_dataset import get_dataloaders
from models.architectures import get_model
from utils.metrics import calculate_metrics

def train():
    # Load config
    with open('configs/train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataloaders
    train_loader, val_loader, test_loader, idx_to_species = get_dataloaders(
        config['dataset']['csv_path'],
        batch_size=config['training']['batch_size'],
        img_size=config['training']['img_size']
    )
    
    num_classes = len(idx_to_species)
    print(f"Number of classes: {num_classes}")
    
    # Model
    model = get_model(num_classes, model_name=config['training']['model_name']).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # Mixed precision
    scaler = GradScaler(enabled=config['training']['mixed_precision'])
    
    # Checkpoint dir
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=config['training']['mixed_precision']):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            acc1, acc5 = calculate_metrics(outputs, labels, topk=(1, 5))
            train_total += labels.size(0)
            train_correct += (acc1.item() * labels.size(0) / 100.0) # Approx, but calculate_metrics returns %
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'top1': f"{acc1.item():.2f}%", 'top5': f"{acc5.item():.2f}%"})
            
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * train_correct / train_total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        env_metrics = {} # environment -> {'correct': 0, 'total': 0}
        
        with torch.no_grad():
            for images, labels, envs in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                acc1, acc5 = calculate_metrics(outputs, labels, topk=(1, 5))
                val_total += labels.size(0)
                val_correct += (acc1.item() * labels.size(0) / 100.0)

                # Track per-environment accuracy
                _, predicted = torch.max(outputs.data, 1)
                for i in range(len(envs)):
                    env = envs[i]
                    if env not in env_metrics:
                        env_metrics[env] = {'correct': 0, 'total': 0}
                    env_metrics[env]['total'] += 1
                    if predicted[i] == labels[i]:
                        env_metrics[env]['correct'] += 1
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = 100 * val_correct / val_total
        
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        print("Environment Robustness (Val Acc):")
        for env, metrics in env_metrics.items():
            acc = 100 * metrics['correct'] / metrics['total']
            print(f"  {env}: {acc:.2f}% ({metrics['correct']}/{metrics['total']})")
        
        scheduler.step(epoch_val_loss)
        
        # Checkpointing
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_acc,
                'idx_to_species': idx_to_species
            }, os.path.join(save_dir, 'best_model.pth'))
            print("Checkpoint saved!")
        else:
            patience_counter += 1
            if patience_counter >= config['training']['patience']:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    # Ensure utils.metrics exists or create a simple one
    os.makedirs('utils', exist_ok=True)
    if not os.path.exists('utils/metrics.py'):
        with open('utils/metrics.py', 'w') as f:
            f.write("def calculate_metrics(outputs, labels):\n    pass\n")
            
    train()
