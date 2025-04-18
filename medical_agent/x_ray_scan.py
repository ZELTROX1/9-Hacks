import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# Define the CheXNet model based on DenseNet121
class CheXNet(nn.Module):
    def __init__(self, num_classes=14):
        super(CheXNet, self).__init__()
        self.densenet121 = models.densenet121(weights="IMAGENET1K_V1")
        num_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.densenet121(x)

# Initialize the model
model = CheXNet(num_classes=14)

# Load pretrained weights if available (you would need a .pth file)
# model.load_state_dict(torch.load("chexnet.pth", map_location=torch.device('cpu')))
# model.eval()

# Transformation for X-ray images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# List of disease classes from CheXNet paper
DISEASE_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

# Function to predict diseases from an image
def predict_xray(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs).squeeze(0)  # Multi-label classification
        prob_dict = {DISEASE_CLASSES[i]: float(probs[i]) for i in range(len(DISEASE_CLASSES))}
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

    return sorted_probs

# Example usage (if needed)
if __name__ == "__main__":
    with open("images.jpeg", "rb") as f:
        image_bytes = f.read()
    predictions = predict_xray(image_bytes)
    for disease, prob in predictions.items():
        print(f"{disease}: {prob:.4f}")
