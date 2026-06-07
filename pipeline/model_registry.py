"""Load and cache ML models once at application startup."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchvision import models

from pipeline.config import (
    LANDMARK_MODEL,
    NECK_LINE_MODEL,
    SEGMENTATION_MODEL,
    SEG_NUM_CLASSES,
)


class PipelineModelError(RuntimeError):
    """Raised when a required model cannot be loaded."""


class FemoralNeckSlopeModel(nn.Module):
    """ResNet34 slope regression model outputting [cos(2θ), sin(2θ)]."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)


class HeatmapUNet(nn.Module):
    """Dual-input U-Net producing per-landmark heatmaps."""

    def __init__(self, in_channels: int = 2, num_landmarks: int = 5) -> None:
        super().__init__()

        def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_channels, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.enc4 = conv_block(128, 256)
        self.center = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block(64, 32)
        self.out_conv = nn.Conv2d(32, num_landmarks, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(c), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _clean_state_dict(state_dict: dict) -> dict:
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise PipelineModelError(f"Model file not found: {path}")
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


class ModelRegistry:
    """Thread-safe singleton holding loaded models."""

    _instance: Optional["ModelRegistry"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.device = _device()
        self._segmentation: Optional[smp.UnetPlusPlus] = None
        self._neck_line: Optional[FemoralNeckSlopeModel] = None
        self._landmarks: Optional[HeatmapUNet] = None
        self._loaded = False

    @classmethod
    def get(cls) -> "ModelRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = ModelRegistry()
            return cls._instance

    def ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._segmentation = self._load_segmentation()
            self._neck_line = self._load_neck_line()
            self._landmarks = self._load_landmarks()
            self._loaded = True

    def _load_segmentation(self) -> smp.UnetPlusPlus:
        model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=1,
            classes=SEG_NUM_CLASSES,
            activation=None,
        )
        checkpoint = _load_checkpoint(SEGMENTATION_MODEL, self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(_clean_state_dict(state_dict))
        model.to(self.device).eval()
        return model

    def _load_neck_line(self) -> FemoralNeckSlopeModel:
        model = FemoralNeckSlopeModel().to(self.device)
        checkpoint = _load_checkpoint(NECK_LINE_MODEL, self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(_clean_state_dict(state_dict), strict=True)
        model.eval()
        return model

    def _load_landmarks(self) -> HeatmapUNet:
        model = HeatmapUNet(in_channels=2, num_landmarks=5).to(self.device)
        checkpoint = _load_checkpoint(LANDMARK_MODEL, self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model

    @property
    def segmentation(self) -> smp.UnetPlusPlus:
        self.ensure_loaded()
        assert self._segmentation is not None
        return self._segmentation

    @property
    def neck_line(self) -> FemoralNeckSlopeModel:
        self.ensure_loaded()
        assert self._neck_line is not None
        return self._neck_line

    @property
    def landmarks(self) -> HeatmapUNet:
        self.ensure_loaded()
        assert self._landmarks is not None
        return self._landmarks
