"""
Day/Night Classifier
---------------------
Path   : classifiers/day_night/main.py
Model  : MobileNetV2  (ImageNet pretrained, frozen backbone)
Input  : BGR frame  (numpy array from OpenCV)
Output : {
    "label":                "day" | "night",
    "confidence":           float,
    "route_to_enhancement": bool,
    "method":               "heuristic" | "cnn"
}

Strategy — two-stage:
  1. Brightness heuristic  →  instant, no GPU, handles obvious day/night
  2. MobileNetV2 CNN       →  runs only for ambiguous twilight / mixed lighting
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Optional


# ─── Model ────────────────────────────────────────────────────────────────────

class DayNightModel(nn.Module):
    """
    MobileNetV2 backbone + custom binary head.
    Backbone: ImageNet pretrained (frozen by default).
    Head:     1280 → 256 → 2  (day / night logits)
    """

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        backbone      = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        """Call when you have enough labelled data for full fine-tuning."""
        for p in self.features.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ─── Inference ────────────────────────────────────────────────────────────────

class DayNightClassifier:
    """
    Public inference class used by pipeline.py.

    Usage:
        classifier = DayNightClassifier()
        result     = classifier.predict(frame)   # frame = BGR numpy array
    """

    LABELS        = ["day", "night"]
    BRIGHT_THRESH = 160   # mean pixel value → definitely day, skip enhancement
    DARK_THRESH   = 85    # ↑ from 60: mean < 85 → route to full enhancement chain
                          #   Previously 60 left a gap: frames with mean 61–85
                          #   (indoor, shade, overcast, golden hour) hit the CNN
                          #   which is untrained and often classifies them as 'day'
                          #   → no enhancement → RetinaFace misses faces in shadow.

    def __init__(
        self,
        model_path : Optional[str] = None,
        device     : str = "auto",
    ):
        self.device = self._resolve_device(device)
        self.model  = DayNightModel(freeze_backbone=True).to(self.device)
        self.model.eval()

        if model_path:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[DayNight] Loaded weights: {model_path}")
        else:
            print(f"[DayNight] Using ImageNet pretrained backbone  ({self.device})")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])

    # ── Public ────────────────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> dict:
        """
        Classify a single BGR frame as day or night.

        Returns:
            label                : "day" | "night"
            confidence           : 0.0 – 1.0
            route_to_enhancement : True when night (pipeline uses this flag)
            method               : "heuristic" | "cnn"
        """
        result = self._brightness_heuristic(frame)
        if result is not None:
            return result
        return self._cnn_predict(frame)

    def save_weights(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"[DayNight] Weights saved → {path}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _brightness_heuristic(self, frame: np.ndarray) -> Optional[dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(np.mean(gray))
        if mean >= self.BRIGHT_THRESH:
            return self._result("day",   min(mean / 255, 0.99), "heuristic")
        if mean <= self.DARK_THRESH:
            return self._result("night", 1.0 - mean / 255,      "heuristic")
        return None

    def _cnn_predict(self, frame: np.ndarray) -> dict:
        img    = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
        day_p, night_p = probs[0].item(), probs[1].item()
        label  = "night" if night_p > day_p else "day"
        return self._result(label, max(day_p, night_p), "cnn")

    @staticmethod
    def _result(label: str, confidence: float, method: str) -> dict:
        return {
            "label":                label,
            "confidence":           round(confidence, 4),
            "route_to_enhancement": label == "night",
            "method":               method,
        }

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)


# ─── Fine-Tune Helper ─────────────────────────────────────────────────────────

def fine_tune(
    train_dir      : str,       # must contain  day/  and  night/  subfolders
    model_path_out : str,
    epochs         : int   = 10,
    lr             : float = 1e-3,
):
    """
    Fine-tune on your own labelled frames.

        train_dir/
            day/    *.jpg
            night/  *.jpg

    Usage:
        from classifiers.day_night.main import fine_tune
        fine_tune("data/day_night", "weights/day_night.pt", epochs=15)
    """
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    loader    = DataLoader(ImageFolder(train_dir, transform), batch_size=32,
                           shuffle=True, num_workers=4)
    model     = DayNightModel(freeze_backbone=False).to(device)
    opt       = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        loss_sum, correct, total = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            correct  += (out.argmax(1) == labels).sum().item()
            total    += labels.size(0)
        print(f"Epoch {epoch+1}/{epochs}  "
              f"loss={loss_sum/len(loader):.4f}  acc={correct/total*100:.1f}%")

    torch.save(model.state_dict(), model_path_out)
    print(f"[DayNight] Saved → {model_path_out}")