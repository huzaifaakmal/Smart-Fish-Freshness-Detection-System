import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.fish_dataset import get_dataloaders
from models.inference import FishInference

def evaluate_model(model_path, csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize inference engine
    engine = FishInference(model_path, device=device)
    
    # Load test dataloader
    _, _, test_loader, idx_to_species = get_dataloaders(csv_path)
    
    y_true = []
    y_pred = []
    environments = []
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for images, labels, envs in test_loader:
            images = images.to(device)
            outputs = engine.model(images)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            environments.extend(envs)
            
    # 1. Classification Report
    report = classification_report(y_true, y_pred, target_names=[idx_to_species[i] for i in range(len(idx_to_species))], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    # 2. Confusion Matrix (Top 30 classes for readability)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 16))
    # For large number of classes, showing top 30 by sample count
    sns.heatmap(cm[:30, :30], annot=True, fmt='d', cmap='Blues', 
                xticklabels=[idx_to_species[i] for i in range(30)], 
                yticklabels=[idx_to_species[i] for i in range(30)])
    plt.title('Confusion Matrix (Top 30 Classes)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_top30.png'))
    plt.close()
    
    # 3. Environment-wise Analysis
    env_results = pd.DataFrame({
        'true': y_true,
        'pred': y_pred,
        'environment': environments
    })
    env_results['correct'] = env_results['true'] == env_results['pred']
    
    env_acc = env_results.groupby('environment')['correct'].mean() * 100
    plt.figure(figsize=(10, 6))
    sns.barplot(x=env_acc.index, y=env_acc.values)
    plt.title('Accuracy by Environment (%)')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(output_dir, 'accuracy_by_env.png'))
    plt.close()
    
    print(f"Evaluation results saved to {output_dir}")
    print("\nSummary Accuracy by Environment:")
    print(env_acc)

if __name__ == "__main__":
    MODEL_PATH = "models/checkpoints/best_model.pth"
    CSV_PATH = "dataset/fish_metadata.csv"
    OUTPUT_DIR = "notebooks/evaluation_results"
    
    if os.path.exists(MODEL_PATH):
        evaluate_model(MODEL_PATH, CSV_PATH, OUTPUT_DIR)
    else:
        print(f"Error: Model not found at {MODEL_PATH}")
