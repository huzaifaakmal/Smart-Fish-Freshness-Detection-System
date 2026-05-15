import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes, model_name='efficientnet_b3', pretrained=True):
    if model_name == 'efficientnet_b3':
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        # Replace the classifier
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'mobilenet_v3':
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
        
    return model

if __name__ == "__main__":
    # Test model creation
    model = get_model(num_classes=297)
    print(f"Model created with {297} classes.")
    # Dummy input
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"Output shape: {y.shape}")
