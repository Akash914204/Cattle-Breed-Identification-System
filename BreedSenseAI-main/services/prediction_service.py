import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
import os


class PredictionService:
    def __init__(self, model_path, num_classes):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------------- LOAD CLASSES ----------------
        classes_path = os.path.join("models", "classes.txt")
        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.num_classes = len(self.class_names)

        # ---------------- LOAD MODEL ----------------
        self.model = resnet18(pretrained=False)   # ✅ SAME AS TRAINING

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            self.num_classes
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        # ✅ IMPORTANT FIX
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)

        self.model.to(self.device)
        self.model.eval()

        # ---------------- TRANSFORM ----------------
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),   # ✅ SAME AS TRAINING
            transforms.ToTensor()
        ])

    def predict(self, image_path):

        if not os.path.exists(image_path):
            raise FileNotFoundError("Image not found")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

        return {
            "breed": self.class_names[pred.item()],
            "confidence": round(confidence.item() * 100, 2)
        }