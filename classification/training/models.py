"""
Model architectures and builders for multi-class classification (Stage 2 pipeline).

Supports: EfficientNet-B0/B2, ConvNeXt-Tiny, ResNet50, MobileNet-V3, Swin-Tiny

Each model:
- Accepts 1-channel SAR imagery input (grayscale)
- Outputs 5-class logits; the controlled trainer assigns them in the order
  Oil, Look-alike, Land, Sea, Ship.
- Uses pre-trained ImageNet weights (first layer adapted for 1-channel input)
- Suitable for multi-hot binary classification via BCEWithLogitsLoss
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0 as tv_efficientnet_b0,
    efficientnet_b2 as tv_efficientnet_b2,
    convnext_tiny as tv_convnext_tiny,
    resnet50 as tv_resnet50,
    mobilenet_v3_large as tv_mobilenet_v3_large,
    swin_t as tv_swin_t,
    EfficientNet_B0_Weights, EfficientNet_B2_Weights,
    ConvNeXt_Tiny_Weights,
    ResNet50_Weights,
    MobileNet_V3_Large_Weights,
    Swin_T_Weights,
)


def _adapt_first_conv_layer(model: nn.Module, in_channels: int = 1, model_type: str = "standard") -> None:
    """
    Adapt the first convolutional layer to accept 1-channel input instead of 3.

    Args:
        model: The model to adapt
        in_channels: Number of input channels (default: 1 for SAR)
        model_type: Type of model architecture ("standard", "convnext", "swin")
    """
    if model_type == "standard":  # EfficientNet, ResNet, MobileNet
        old_conv = model.features[0][0] if hasattr(model, 'features') else model.conv1

        if hasattr(model, 'features') and isinstance(model.features[0], nn.Sequential):
            # EfficientNet style
            old_conv = model.features[0][0]
        else:
            # ResNet/MobileNet style
            old_conv = model.conv1

        # Create new conv layer with 1 input channel
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        # Average the pretrained weights across the 3 RGB channels to adapt for 1 channel
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        # Replace the old conv layer
        if hasattr(model, 'features') and isinstance(model.features[0], nn.Sequential):
            model.features[0][0] = new_conv
        else:
            model.conv1 = new_conv

    elif model_type == "convnext":
        # ConvNeXt: adapt stem layer (first conv in stem)
        old_conv = model.features[0][0]  # First conv in stem

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        model.features[0][0] = new_conv

    elif model_type == "swin":
        # Swin: adapt patch embed layer (first conv in features[0])
        old_conv = model.features[0][0]  # patch embedding conv

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        model.features[0][0] = new_conv


def effnet_b0(num_classes: int = 5) -> nn.Module:
    """
    EfficientNet-B0 for 1-channel SAR multi-class classification.

    EfficientNet-B0 is a lightweight, efficient architecture combining
    compound scaling of depth, width, and resolution. Pre-trained on ImageNet.

    Adaptation: First convolutional layer modified from 3-channel RGB to 1-channel
    SAR imagery input via weight averaging.

    Args:
        num_classes: Number of output classes (default: 5)

    Returns:
        nn.Module: EfficientNet-B0 with 1-channel input and num_classes output
    """
    model = tv_efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    _adapt_first_conv_layer(model, in_channels=1, model_type="standard")

    # Replace final classifier
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

    return model


def effnet_b2(num_classes: int = 5) -> nn.Module:
    """
    EfficientNet-B2 for 1-channel SAR multi-class classification.

    EfficientNet-B2 builds on the EfficientNet scaling principles with
    increased depth and width compared to B0. Pre-trained on ImageNet.

    Adaptation: First convolutional layer modified from 3-channel RGB to 1-channel
    SAR imagery input via weight averaging.

    Args:
        num_classes: Number of output classes (default: 5)

    Returns:
        nn.Module: EfficientNet-B2 with 1-channel input and num_classes output
    """
    model = tv_efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    _adapt_first_conv_layer(model, in_channels=1, model_type="standard")

    # Replace final classifier
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

    return model


def convnext_tiny(num_classes: int = 5) -> nn.Module:
    """
    ConvNeXt-Tiny for 1-channel SAR multi-class classification.

    ConvNeXt is a modern pure convolutional architecture modernizing ResNet
    with design choices from vision transformers. The Tiny variant is efficient
    while maintaining strong performance. Pre-trained on ImageNet.

    Adaptation: Patch embedding (first stem convolution) modified from 3-channel
    RGB to 1-channel SAR imagery input via weight averaging.

    Args:
        num_classes: Number of output classes (default: 5)

    Returns:
        nn.Module: ConvNeXt-Tiny with 1-channel input and num_classes output
    """
    model = tv_convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    _adapt_first_conv_layer(model, in_channels=1, model_type="convnext")

    # Replace final classifier
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, num_classes)

    return model


def resnet50(num_classes: int = 5) -> nn.Module:
    """
    ResNet50 for 1-channel SAR multi-class classification.

    ResNet50 is a deep residual network with 50 layers using skip connections
    to address vanishing gradient problems. Pre-trained on ImageNet (V2 weights).

    Adaptation: First convolutional layer modified from 3-channel RGB to 1-channel
    SAR imagery input via weight averaging.

    Args:
        num_classes: Number of output classes (default: 5)

    Returns:
        nn.Module: ResNet50 with 1-channel input and num_classes output
    """
    model = tv_resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    _adapt_first_conv_layer(model, in_channels=1, model_type="standard")

    # Replace final classifier
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def mobilenet_v3_large(num_classes: int = 5) -> nn.Module:
    """
    MobileNet-V3-Large for 1-channel SAR multi-class classification.

    MobileNet-V3-Large is an efficient mobile architecture using inverted
    residuals and linear bottlenecks with automated neural architecture search.
    Pre-trained on ImageNet (V2 weights).

    Adaptation: First convolutional layer modified from 3-channel RGB to 1-channel
    SAR imagery input via weight averaging.

    Args:
        num_classes: Number of output classes (default: 5)

    Returns:
        nn.Module: MobileNet-V3-Large with 1-channel input and num_classes output
    """
    model = tv_mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    _adapt_first_conv_layer(model, in_channels=1, model_type="standard")

    # Replace final classifier
    num_features = model.classifier[0].in_features
    model.classifier[0] = nn.Linear(num_features, 1280)
    model.classifier[3] = nn.Linear(1280, num_classes)

    return model


def swin_t(num_classes: int = 5) -> nn.Module:
    """
    Swin-Tiny (Shifted Window Transformer) for 1-channel SAR multi-class classification.

    Swin Transformer is a hierarchical vision transformer using shifted windows
    for efficient attention computation. The Tiny variant provides a good balance
    between efficiency and performance. Pre-trained on ImageNet.

    Adaptation: Patch embedding layer modified from 3-channel RGB to 1-channel
    SAR imagery input via weight averaging.

    Args:
        num_classes: Number of output classes (default: 5)

    Returns:
        nn.Module: Swin-Tiny with 1-channel input and num_classes output
    """
    model = tv_swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    _adapt_first_conv_layer(model, in_channels=1, model_type="swin")

    # Replace final classifier
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, num_classes)

    return model


class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)
        return torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))


class CBAM(nn.Module):
    """Convolutional block attention module (channel then spatial), as in the first-round Stage-2 code."""
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel_attn = _ChannelAttention(channels, reduction)
        self.spatial_attn = _SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attn(x)
        return x * self.spatial_attn(x)


class _SpatialTransformer(nn.Module):
    """Transformer encoder over the final feature map; returns a map of the same shape.

    Sizes follow the first-round code (3 layers, 8 heads, width 256, MLP 512, dropout 0.1,
    learned position embedding for up to 1024 tokens). A residual projection back to the
    backbone width keeps the shared pooling and linear head identical to the control.
    """
    def __init__(self, channels: int, embed_dim: int = 256, heads: int = 8, layers: int = 3,
                 mlp_dim: int = 512, dropout: float = 0.1, max_tokens: int = 1024) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.proj_in = nn.Linear(channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Linear(embed_dim, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        if n > self.max_tokens:
            raise ValueError(f"Token count {n} exceeds max_tokens={self.max_tokens}")
        tokens = self.proj_in(x.flatten(2).transpose(1, 2)) + self.pos_embed[:, :n]
        tokens = self.norm(self.encoder(tokens))
        return x + self.proj_out(tokens).transpose(1, 2).reshape(b, c, h, w)


class EffNetAttention(nn.Module):
    """EfficientNet-B2 with CBAM (and optionally a Transformer) on the final feature map.

    Pooling and the linear classifier are those of the plain EfficientNet-B2, so the only
    change against the control is the inserted attention. `head_module` groups the new
    modules with the classifier so the trainer treats them as head parameters (trainable
    during the adaptation epochs); `features` and `classifier` keep the torchvision names.
    """
    def __init__(self, num_classes: int = 5, transformer: bool = False) -> None:
        super().__init__()
        base = effnet_b2(num_classes=num_classes)
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = base.classifier
        channels = self.classifier[1].in_features
        self.cbam = CBAM(channels)
        self.transformer = _SpatialTransformer(channels) if transformer else None
        extra = [self.cbam] + ([self.transformer] if transformer else [])
        # Not registered as a child (all members already are), so state_dict keys stay unique.
        object.__setattr__(self, 'head_module', nn.ModuleList([self.classifier] + extra))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cbam(self.features(x))
        if self.transformer is not None:
            x = self.transformer(x)
        return self.classifier(torch.flatten(self.avgpool(x), 1))


def effnet_b2_cbam(num_classes: int = 5) -> nn.Module:
    """EfficientNet-B2 + CBAM (configuration A of the 2026-10-18 controlled attention test)."""
    return EffNetAttention(num_classes, transformer=False)


def effnet_b2_cbam_transformer(num_classes: int = 5) -> nn.Module:
    """EfficientNet-B2 + CBAM + Transformer (configuration B)."""
    return EffNetAttention(num_classes, transformer=True)


def build_model(model_name: str, num_classes: int = 5) -> nn.Module:
    """
    Build a pre-trained model for 1-channel SAR multi-class classification.

    Supports 6 architecture families, each adapted from 3-channel ImageNet
    pre-training to 1-channel SAR input via weight averaging on the first layer.
    All models output logits suitable for BCEWithLogitsLoss (multi-hot classification).

    Args:
        model_name: Name of the model architecture. Must be one of:
            - 'effnet_b0': EfficientNet-B0 (lightweight)
            - 'effnet_b2': EfficientNet-B2 (balanced)
            - 'convnext_tiny': ConvNeXt-Tiny (modern conv)
            - 'resnet50': ResNet50 (deep residual)
            - 'mobilenet_v3_large': MobileNet-V3-Large (efficient)
            - 'swin_tiny': Swin-Tiny (transformer)
        num_classes: Number of output classes (default: 5)

    Returns:
        nn.Module: A PyTorch model with the requested architecture

    Raises:
        ValueError: If model_name is not recognized
    """
    models_dict = {
        'effnet_b0': effnet_b0,
        'effnet_b2': effnet_b2,
        'convnext_tiny': convnext_tiny,
        'resnet50': resnet50,
        'mobilenet_v3_large': mobilenet_v3_large,
        'swin_tiny': swin_t,
        'effnet_b2_cbam': effnet_b2_cbam,
        'effnet_b2_cbam_transformer': effnet_b2_cbam_transformer,
    }

    if model_name not in models_dict:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Choose from: {list(models_dict.keys())}"
        )

    return models_dict[model_name](num_classes=num_classes)
