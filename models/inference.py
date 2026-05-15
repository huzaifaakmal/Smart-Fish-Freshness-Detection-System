import torch
from torchvision import transforms
from PIL import Image
import os
import sys

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.architectures import get_model

class FishInference:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.idx_to_species = checkpoint['idx_to_species']
        num_classes = len(self.idx_to_species)
        
        # Initialize model (assuming efficientnet_b3 for now, can be parameterized)
        self.model = get_model(num_classes, model_name='efficientnet_b3').to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def predict(self, image_path, topk=5):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            top_prob, top_idx = torch.topk(probabilities, topk)
            
        results = []
        for i in range(topk):
            results.append({
                "species": self.idx_to_species[str(top_idx[0][i].item())] if isinstance(self.idx_to_species, dict) and str(top_idx[0][i].item()) in self.idx_to_species else self.idx_to_species[top_idx[0][i].item()],
                "confidence": float(top_prob[0][i].item())
            })
            
        return {
            "predicted_species": results[0]["species"],
            "confidence": results[0]["confidence"],
            "top_5": results
        }

if __name__ == "__main__":
    # This is a placeholder since we don't have a trained model yet
    print("Inference engine ready. (Requires trained model checkpoint)")
