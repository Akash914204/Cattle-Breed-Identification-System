"""
gradcam_service.py — Grad-CAM Heatmap Generator
=================================================
Custom implementation of Gradient-weighted Class Activation Mapping (Grad-CAM)
for visualising which regions of a cattle image drive the model's breed prediction.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
"""

import os
import uuid
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """
    Generate Grad-CAM heatmaps for a given CNN model.

    Parameters
    ----------
    model : torch.nn.Module
        A classification model in eval mode.
    target_layer : torch.nn.Module
        The convolutional layer to hook into (usually the last conv layer).
    device : torch.device
        CPU or CUDA device.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module,
                 device: torch.device | None = None):
        self.model = model
        self.target_layer = target_layer
        self.device = device or torch.device("cpu")

        # Storage for hooked tensors
        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None

        # Register forward & backward hooks
        self._fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _forward_hook(self, module, input, output):
        """Capture activations from the target layer."""
        self._activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Capture gradients flowing into the target layer."""
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, input_tensor: torch.Tensor,
                 class_idx: int | None = None) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for a single image tensor.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Pre-processed image tensor of shape [1, C, H, W].
        class_idx : int, optional
            Target class index. If ``None``, the predicted class is used.

        Returns
        -------
        np.ndarray
            Heatmap of shape (H, W) with values in [0, 1].
        """
        self.model.zero_grad()
        input_tensor = input_tensor.to(self.device).requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)                # [1, num_classes]

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass for the target class score
        target_score = output[0, class_idx]
        target_score.backward()

        # Grad-CAM computation
        gradients = self._gradients                       # [1, C, h, w]
        activations = self._activations                   # [1, C, h, w]

        # Global-average-pool the gradients → channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)                                 # ReLU to keep positive

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() != cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        """Clean up registered hooks."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()


# ---------------------------------------------------------------------------
# Overlay helper
# ---------------------------------------------------------------------------

def create_heatmap_overlay(
    original_image_path: str,
    cam: np.ndarray,
    save_dir: str,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> str:
    """
    Overlay a Grad-CAM heatmap on the original image and save to disk.

    Parameters
    ----------
    original_image_path : str
        Path to the original cattle image.
    cam : np.ndarray
        Heatmap array of shape (h, w) with values in [0, 1].
    save_dir : str
        Directory to save the output image.
    alpha : float
        Blending factor for the overlay (0 = only image, 1 = only heatmap).
    colormap : str
        Matplotlib colormap name.

    Returns
    -------
    str
        Filename of the saved heatmap image (relative to save_dir).
    """
    # Load and resize original image
    original = Image.open(original_image_path).convert("RGB")
    width, height = original.size

    # Resize CAM to match original image dimensions
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (width, height), Image.BILINEAR
        )
    ) / 255.0

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(cam_resized)[:, :, :3]          # drop alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend
    original_np = np.array(original).astype(np.float32)
    overlay = (
        (1 - alpha) * original_np + alpha * heatmap_colored.astype(np.float32)
    ).astype(np.uint8)

    # Save
    os.makedirs(save_dir, exist_ok=True)
    heatmap_filename = f"gradcam_{uuid.uuid4().hex[:10]}.png"
    output_path = os.path.join(save_dir, heatmap_filename)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=12, fontweight="bold", color="#e5e7eb")
    axes[0].axis("off")

    axes[1].imshow(cam_resized, cmap=colormap, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12, fontweight="bold", color="#e5e7eb")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=12, fontweight="bold", color="#e5e7eb")
    axes[2].axis("off")

    fig.patch.set_facecolor("#111827")
    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, bbox_inches="tight", dpi=150,
                facecolor="#111827", edgecolor="none")
    plt.close(fig)

    return heatmap_filename


# ---------------------------------------------------------------------------
# Convenience: end-to-end Grad-CAM for a ResNet / EfficientNet model
# ---------------------------------------------------------------------------

def generate_gradcam(model, image_path: str, transform, device, save_dir: str,
                     class_idx: int | None = None) -> str:
    """
    End-to-end Grad-CAM: preprocess → generate CAM → overlay → save.

    Automatically selects the correct target layer based on architecture:
      - ResNet:        model.layer4[-1]
      - EfficientNet:  model.features[-1]

    Returns the heatmap image filename.
    """
    # Auto-detect the last conv layer
    if hasattr(model, "layer4"):                           # ResNet family
        target_layer = model.layer4[-1]
    elif hasattr(model, "features"):                       # EfficientNet / MobileNet
        target_layer = model.features[-1]
    else:
        raise ValueError(
            "Cannot auto-detect target layer. Pass it manually to GradCAM."
        )

    # Prepare input
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    # Run Grad-CAM
    grad_cam = GradCAM(model=model, target_layer=target_layer, device=device)
    try:
        cam = grad_cam.generate(input_tensor, class_idx=class_idx)
    finally:
        grad_cam.remove_hooks()

    # Create overlay image
    heatmap_filename = create_heatmap_overlay(
        original_image_path=image_path,
        cam=cam,
        save_dir=save_dir,
        alpha=0.5,
    )

    return heatmap_filename
