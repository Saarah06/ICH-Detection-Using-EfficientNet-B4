import torch
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import pydicom
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter  # For smoothing

# -------------------------
# 1. Device & Model (unchanged)
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=6).to(device)
model.eval()

labels = ["Epidural", "Intraparenchymal", "Intraventricular", "Subarachnoid", "Subdural", "Any"]

# -------------------------
# 2. Load & Process DICOM (added ImageNet normalization)
# -------------------------
dicom_path = "sample_ct.dcm"  # Update path as needed
ds = pydicom.dcmread(dicom_path)
img = ds.pixel_array.astype(np.float32)

if hasattr(ds, 'RescaleSlope'):
    img = img * ds.RescaleSlope + ds.RescaleIntercept

# Windowing for Hemorrhage (L=80, W=200)
wc, ww = 80, 200 
img = np.clip(img, wc - ww//2, wc + ww//2)
img = (img - (wc - ww//2)) / ww  # Now [0,1]

# Resize to EfficientNet-B4 input: 380x380
display_img = cv2.resize(np.stack([img]*3, axis=-1), (380, 380))

# ImageNet normalization (CRITICAL for pretrained model)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
input_tensor = torch.from_numpy((display_img - mean) / std).permute(2,0,1).unsqueeze(0).float().to(device)

# -------------------------
# 3. Prediction (use softmax for multi-label if needed, but argmax top-5)
# -------------------------
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.sigmoid(output).cpu().numpy()[0]  # Multi-label sigmoid
best_idx = np.argmax(probs[:5])  # Top hemorrhage type (exclude 'Any')

# -------------------------
# 4. Improved Grad-CAM
# -------------------------
# Better target: very last conv block in EfficientNet-B4 (blocks[7][-1] for B4 specifically)
target_layers = [model.blocks[-1]]  # NEW - last block (safe & optimal for B4)


cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(best_idx)])[0]

# --- BETTER REFINEMENTS ---
# 1. Lower percentile threshold (85%) + Gaussian smooth (sigma=2) for focused/natural heatmap
threshold = np.percentile(grayscale_cam, 85)
grayscale_cam = np.maximum(grayscale_cam - threshold, 0)  # Subtract for sharper edges
grayscale_cam = gaussian_filter(grayscale_cam, sigma=2)  # Smooth artifacts[web:3]

# 2. Tighter brain mask (mean > 0.05, dilate slightly)
mask = cv2.dilate((display_img.mean(axis=-1) > 0.05).astype(np.float32), np.ones((5,5), np.uint8))
grayscale_cam *= mask

# 3. Renormalize
grayscale_cam = grayscale_cam / grayscale_cam.max() if grayscale_cam.max() > 0 else grayscale_cam

# -------------------------
# 5. Visualization (higher image_weight for clearer overlay)
# -------------------------
vis = show_cam_on_image(display_img, grayscale_cam, use_rgb=True, image_weight=0.6)

plt.figure(figsize=(15, 6))
plt.subplot(1, 3, 1)
plt.imshow(display_img)
plt.title("Original CT (Hemorrhage Window)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(grayscale_cam, cmap='jet')
plt.title("Raw Grad-CAM Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(vis)
plt.title(f"Grad-CAM: {labels[best_idx]} (prob: {probs[best_idx]:.3f})")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save improved output
cv2.imwrite("improved_gradcam.jpg", cv2.cvtColor((vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
print(f"Top predictions: {dict(zip(labels, np.round(probs, 3)))}")  # Print all probs
