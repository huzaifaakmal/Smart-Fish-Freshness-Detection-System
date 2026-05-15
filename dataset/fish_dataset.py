import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os

class FishDataset(Dataset):
    def __init__(self, metadata_df, transform=None):
        self.df = metadata_df
        self.transform = transform
        
        # Create class mapping
        self.species_to_idx = {name: i for i, name in enumerate(sorted(self.df['species_name'].unique()))}
        self.idx_to_species = {i: name for name, i in self.species_to_idx.items()}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        species_name = row['species_name']
        environment = row['environment']
        
        image = Image.open(image_path).convert("RGB")
        label = self.species_to_idx[species_name]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, environment
def get_dataloaders(csv_path, batch_size=32, img_size=224):
    df = pd.read_csv(csv_path)

    # Filter out classes with too few samples for stratified 3-way split
    # We need at least enough samples to have at least one in each split (train, val, test)
    # and for stratification to work in both splits.
    counts = df['species_name'].value_counts()
    valid_species = counts[counts >= 7].index
    if len(valid_species) < len(counts):
        print(f"Warning: Filtering out {len(counts) - len(valid_species)} species with < 7 samples.")
        df = df[df['species_name'].isin(valid_species)].reset_index(drop=True)

    # Stratified split: Train (70%), Test+Val (30%)
    train_df, test_val_df = train_test_split(
        df, test_size=0.3, stratify=df['species_name'], random_state=42
    )
    # Split Test+Val into Val (15%) and Test (15%)
    val_df, test_df = train_test_split(
        test_val_df, test_size=0.5, stratify=test_val_df['species_name'], random_state=42
    )

    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds = FishDataset(train_df, transform=train_transform)
    val_ds = FishDataset(val_df, transform=val_test_transform)
    test_ds = FishDataset(test_df, transform=val_test_transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_ds.idx_to_species

if __name__ == "__main__":
    # Test loading
    CSV_PATH = "dataset/fish_metadata.csv"
    try:
        train_loader, val_loader, test_loader, idx_to_species = get_dataloaders(CSV_PATH)
        print(f"Dataset Split: {len(train_loader.dataset)} Train, {len(val_loader.dataset)} Val, {len(test_loader.dataset)} Test")
        print(f"Sample Species Mapping: {list(idx_to_species.items())[:5]}")
        
        # Check a batch
        images, labels, envs = next(iter(train_loader))
        print(f"Batch Image Shape: {images.shape}")
        print(f"Batch Labels: {labels[:5]}")
        print(f"Batch Envs: {envs[:5]}")
    except Exception as e:
        print(f"Error: {e}")
