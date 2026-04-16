# """
# inference.py — Cattle Breed Prediction Module
# ==============================================
# Loads a trained ResNet-50 model and exposes a `predict_breed()` function
# that accepts an image path, preprocesses the image, runs inference, and
# returns the predicted breed name with a confidence percentage.

# Usage:
#     from inference import predict_breed

#     result = predict_breed("path/to/cattle_image.jpg")
#     print(result)  # {"breed": "Gir", "confidence": 97.42}
# """

# import os
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision.models import resnet50
# from PIL import Image, UnidentifiedImageError

# # ---------------------------------------------------------------------------
# # Constants
# # ---------------------------------------------------------------------------

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pth")
# CLASSES_PATH = os.path.join(BASE_DIR, "models", "classes.txt")

# # The 5 Indian cattle breeds (fallback if classes.txt is missing)
# DEFAULT_CLASSES = ["Gir", "Sahiwal", "Red Sindhi", "Tharparkar", "Kankrej"]

# NUM_CLASSES = 5

# # ---------------------------------------------------------------------------
# # Preprocessing Pipeline
# # ---------------------------------------------------------------------------

# preprocess = transforms.Compose([
#     transforms.Resize(256),                         # Resize shortest edge to 256
#     transforms.CenterCrop(224),                     # Center-crop to 224×224
#     transforms.ToTensor(),                          # Convert to tensor [0, 1]
#     transforms.Normalize(                           # ImageNet normalisation
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
# ])

# # ---------------------------------------------------------------------------
# # Model Loading
# # ---------------------------------------------------------------------------

# def _load_class_names() -> list[str]:
#     """Load breed names from models/classes.txt (one name per line)."""
#     if os.path.exists(CLASSES_PATH):
#         with open(CLASSES_PATH, "r", encoding="utf-8") as f:
#             names = [line.strip() for line in f if line.strip()]
#             if names:
#                 return names
#     return DEFAULT_CLASSES


# def _load_model(model_path: str) -> nn.Module:
#     """Build a ResNet-50 with the correct final layer and load weights."""
#     model = resnet50(weights=None)

#     # Replace the final fully-connected layer to match our 5-class output
#     in_features = model.fc.in_features          # 2048 for ResNet-50
#     model.fc = nn.Linear(in_features, NUM_CLASSES)

#     if not os.path.exists(model_path):
#         raise FileNotFoundError(
#             f"Model weights not found at '{model_path}'. "
#             "Place your trained model.pth in the models/ directory."
#         )

#     # Load weights (CPU-safe regardless of where the model was trained)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     state_dict = torch.load(model_path, map_location=device, weights_only=True)
#     model.load_state_dict(state_dict)

#     model.to(device)
#     model.eval()
#     return model


# # Lazy-loaded globals (initialised on first call to predict_breed)
# _model: nn.Module | None = None
# _class_names: list[str] | None = None
# _device: torch.device | None = None


# def _ensure_model_loaded() -> None:
#     """Initialise the model and class list once."""
#     global _model, _class_names, _device
#     if _model is None:
#         _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         _class_names = _load_class_names()
#         _model = _load_model(MODEL_PATH)

# # ---------------------------------------------------------------------------
# # Public API
# # ---------------------------------------------------------------------------

# def predict_breed(image_path: str) -> dict:
#     """
#     Predict the cattle breed from an image file.

#     Parameters
#     ----------
#     image_path : str
#         Absolute or relative path to a JPEG/PNG image.

#     Returns
#     -------
#     dict
#         {
#             "breed"      : str   – predicted breed name,
#             "confidence" : float – confidence percentage (0–100),
#         }

#     Raises
#     ------
#     FileNotFoundError
#         If `image_path` does not exist.
#     ValueError
#         If the file cannot be opened as a valid image (corrupt / unsupported).
#     RuntimeError
#         If model weights are missing or inference fails unexpectedly.
#     """

#     # --- Validate file existence ---
#     if not os.path.isfile(image_path):
#         raise FileNotFoundError(f"Image not found: '{image_path}'")

#     # --- Attempt to open & validate the image ---
#     try:
#         image = Image.open(image_path)
#         image.verify()                       # Check for corruption
#         image = Image.open(image_path)       # Re-open (verify() closes it)
#         image = image.convert("RGB")         # Ensure 3-channel RGB
#     except (UnidentifiedImageError, OSError, SyntaxError) as exc:
#         raise ValueError(
#             f"Corrupt or unsupported image file: '{image_path}'"
#         ) from exc

#     # --- Load model if not yet initialised ---
#     try:
#         _ensure_model_loaded()
#     except FileNotFoundError:
#         raise  # Propagate missing-model error as-is
#     except Exception as exc:
#         raise RuntimeError(f"Failed to load model: {exc}") from exc

#     # --- Preprocess ---
#     input_tensor = preprocess(image).unsqueeze(0).to(_device)  # [1, 3, 224, 224]

#     # --- Inference ---
#     with torch.no_grad():
#         logits = _model(input_tensor)                          # [1, NUM_CLASSES]
#         probabilities = torch.nn.functional.softmax(logits, dim=1)
#         confidence, predicted_idx = torch.max(probabilities, dim=1)

#     breed_name = _class_names[predicted_idx.item()]
#     confidence_pct = round(confidence.item() * 100, 2)

#     return {
#         "breed": breed_name,
#         "confidence": confidence_pct,
#     }


# # ---------------------------------------------------------------------------
# # Quick CLI test
# # ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) < 2:
#         print("Usage: python inference.py <image_path>")
#         sys.exit(1)

#     path = sys.argv[1]
#     try:
#         result = predict_breed(path)
#         print(f"🐄 Breed     : {result['breed']}")
#         print(f"📊 Confidence: {result['confidence']}%")
#     except (FileNotFoundError, ValueError, RuntimeError) as e:
#         print(f"❌ Error: {e}")
#         sys.exit(1)

"""
inference.py — Cattle Breed Prediction Module (FIXED)
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "weights", "best_model.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "classes.txt")

# ---------------------------------------------------------------------------
# LOAD CLASSES
# ---------------------------------------------------------------------------

def _load_class_names():
    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError("classes.txt not found!")

    with open(CLASSES_PATH, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    return classes

# ---------------------------------------------------------------------------
# TRANSFORM (MATCH TRAINING)
# ---------------------------------------------------------------------------

preprocess = transforms.Compose([
    transforms.Resize((128, 128)),   # SAME as training
    transforms.ToTensor()
])

# ---------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------

def _load_model():
    classes = _load_class_names()
    num_classes = len(classes)

    model = resnet18(pretrained=False)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found!")

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    return model, classes

# ---------------------------------------------------------------------------
# GLOBALS
# ---------------------------------------------------------------------------

_model = None
_classes = None

def _ensure_loaded():
    global _model, _classes
    if _model is None:
        _model, _classes = _load_model()

# ---------------------------------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------------------------------

def predict_breed(image_path):

    # Check file
    if not os.path.exists(image_path):
        raise FileNotFoundError("Image not found!")

    # Load image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        raise ValueError("Invalid image file!")

    # Load model
    _ensure_loaded()

    # Preprocess
    image = preprocess(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = _model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    breed = _classes[predicted.item()]
    confidence = round(confidence.item() * 100, 2)

    return {
        "breed": breed,
        "confidence": confidence
    }

# ---------------------------------------------------------------------------
# TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    path = input("Enter image path: ")
    result = predict_breed(path)
    print(result)