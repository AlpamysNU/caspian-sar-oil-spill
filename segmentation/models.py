"""The Stage-3 segmentation models of the thesis, each described by a SPEC that `train.py` and the inference scripts read.

  segformer_b0_shift     SegFormer-B0 control (the baseline model and recipe on the shared loop; use --shift-aug)
  segformer_b2           SegFormer-B2, a capacity increase over B0 (about 27 M parameters instead of 3.7 M)
  segformer_b2_hrda_c    HRDA-C: one shared B2 run on the 256 px window and on the 512 px scene context reduced to 256 px;
                         the logits are averaged with fixed weights (the model of the thesis)
  segformer_b2_hrda_d    HRDA-D: as HRDA-C with a learned per-pixel, per-class scale attention instead of the fixed average
  deeplabv3plus_resnet50 ResNet-50 + DeepLabV3+, the model of the published Caspian study (HTSM Oil Spill Segmentation,
                         Satyanarayana and Dhali, MIT licence)

All of them take one SAR channel (first convolution replaced by the channel mean), cross-entropy with ignore index 255, AdamW +
cosine schedule. A model is an `nn.Module` with `forward(image, target=None) -> (loss | None, logits)` and `config_dict()`;
a SPEC gives `build(config=None)`, `param_groups(model, lr)`, the default learning rate, and for the two-scale models a
`dataset_factory` that adds the scene context (`context.py`).
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

from segmentation.common import build_segformer, first_conv_to_single_channel
from segmentation.context import ContextPatches

SEGFORMER_B0 = 'nvidia/segformer-b0-finetuned-ade-512-512'
SEGFORMER_B2 = 'nvidia/segformer-b2-finetuned-ade-512-512'
AUX_WEIGHT = 0.5
HEAD_LR_FACTOR = 10
NUM_CLASSES = 5
ASPP_RATES = (12, 24, 36)
RESNET50 = 'torchvision resnet50 IMAGENET1K_V1'




def central_upsample(logits, size):
    """Central half of a (B,C,h,w) logit map (the context view's footprint of the window), bilinear to size x size."""
    h, w = logits.shape[-2:]
    centre = logits[..., h // 4:h - h // 4, w // 4:w - w // 4]
    return F.interpolate(centre, size=size, mode='bilinear', align_corners=False)


class TwoScaleStep(torch.nn.Module):
    """forward(image[B,2,H,W], target=None) -> (loss | None, fused logits at window resolution)."""

    def __init__(self, backbone, aux_weight=AUX_WEIGHT):
        super().__init__()
        self.backbone, self.aux_weight = backbone, aux_weight

    def forward(self, image, target=None):
        size = image.shape[-2:]
        detail = F.interpolate(self.backbone(pixel_values=image[:, :1]).logits, size=size, mode='bilinear', align_corners=False)
        context = central_upsample(self.backbone(pixel_values=image[:, 1:]).logits, size)
        fused = 0.5 * (detail + context)
        loss = None
        if target is not None:
            loss = (F.cross_entropy(fused, target, ignore_index=255)
                    + self.aux_weight * (F.cross_entropy(detail, target, ignore_index=255) + F.cross_entropy(context, target, ignore_index=255)))
        return loss, fused

    def config_dict(self):
        return self.backbone.config.to_dict()


HEAD_LR_FACTOR = 10


class ScaleAttentionStep(torch.nn.Module):
    """forward(image[B,2,H,W], target=None) -> (loss | None, fused logits at window resolution)."""

    def __init__(self, backbone, aux_weight=AUX_WEIGHT):
        super().__init__()
        self.backbone, self.aux_weight = backbone, aux_weight
        self.attention = torch.nn.Sequential(torch.nn.Conv2d(10, 32, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(32, 5, 1))
        torch.nn.init.zeros_(self.attention[2].weight)
        torch.nn.init.zeros_(self.attention[2].bias)

    def forward(self, image, target=None):
        size = image.shape[-2:]
        raw_detail = self.backbone(pixel_values=image[:, :1]).logits
        raw_context = self.backbone(pixel_values=image[:, 1:]).logits
        weight = torch.sigmoid(self.attention(torch.cat([raw_detail, central_upsample(raw_context, raw_detail.shape[-2:])], 1)))
        weight = F.interpolate(weight, size=size, mode='bilinear', align_corners=False)
        detail = F.interpolate(raw_detail, size=size, mode='bilinear', align_corners=False)
        context = central_upsample(raw_context, size)
        fused = (1 - weight) * detail + weight * context
        loss = None
        if target is not None:
            loss = (F.cross_entropy(fused, target, ignore_index=255)
                    + self.aux_weight * (F.cross_entropy(detail, target, ignore_index=255) + F.cross_entropy(context, target, ignore_index=255)))
        return loss, fused

    def config_dict(self):
        return self.backbone.config.to_dict()






def conv_bn_relu(cin, cout, k, dilation=1):
    return torch.nn.Sequential(torch.nn.Conv2d(cin, cout, k, padding=dilation * (k // 2), dilation=dilation, bias=False),
                               torch.nn.BatchNorm2d(cout), torch.nn.ReLU(inplace=True))


class ASPP(torch.nn.Module):
    def __init__(self, cin, cout=256, rates=ASPP_RATES):
        super().__init__()
        self.branches = torch.nn.ModuleList([conv_bn_relu(cin, cout, 1)] + [conv_bn_relu(cin, cout, 3, r) for r in rates])
        self.pool = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), conv_bn_relu(cin, cout, 1))
        self.project = torch.nn.Sequential(conv_bn_relu((len(rates) + 2) * cout, cout, 1), torch.nn.Dropout(0.1))

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        outputs.append(F.interpolate(self.pool(x), size=x.shape[-2:], mode='bilinear', align_corners=False))
        return self.project(torch.cat(outputs, dim=1))


class DeepLabV3PlusResNet50(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        first_conv_to_single_channel(backbone)
        self.stem = torch.nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layers = torch.nn.Sequential(backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4)
        self.aspp = ASPP(2048)
        self.skip = conv_bn_relu(64, 48, 1)
        self.decoder = torch.nn.Sequential(conv_bn_relu(256 + 48, 256, 3), torch.nn.Conv2d(256, NUM_CLASSES, 1))

    def forward(self, image, target=None):
        stem = self.stem(image)
        skip = self.skip(stem)
        high = F.interpolate(self.aspp(self.layers(stem)), size=skip.shape[-2:], mode='bilinear', align_corners=False)
        logits = self.decoder(torch.cat([skip, high], dim=1))
        logits = F.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)
        loss = F.cross_entropy(logits, target, ignore_index=255) if target is not None else None
        return loss, logits

    def config_dict(self):
        return {'backbone': 'resnet50', 'num_classes': NUM_CLASSES, 'aspp_rates': list(ASPP_RATES)}

# ----------------------------------------------------------------------------- specs
def segformer_builder(pretrained):
    return lambda config=None: build_segformer(pretrained, config)


def single_group(model, lr):
    return [{'params': list(model.parameters()), 'lr': lr}]


def two_scale_dataset(frame, data_root, mask_paths, patch_masks, train, shift_max, shift_prob, scenes_root):
    return ContextPatches(frame, data_root, mask_paths, patch_masks, train, shift_max, shift_prob, scenes_root)


def hrda_c_model(config=None):
    return TwoScaleStep(build_segformer(SEGFORMER_B2, config).model)


def hrda_d_model(config=None):
    return ScaleAttentionStep(build_segformer(SEGFORMER_B2, config).model)


def hrda_d_groups(model, lr):
    return [{'params': list(model.backbone.parameters()), 'lr': lr},
            {'params': list(model.attention.parameters()), 'lr': lr * HEAD_LR_FACTOR}]


def deeplab_model(config=None):
    return DeepLabV3PlusResNet50(pretrained=config is None)


def deeplab_groups(model, lr):
    backbone = list(model.stem.parameters()) + list(model.layers.parameters())
    ids = {id(p) for p in backbone}
    head = [p for p in model.parameters() if id(p) not in ids]
    return [{'params': backbone, 'lr': lr}, {'params': head, 'lr': lr * 10}]


_CE = 'cross_entropy_ignore_padding_255'
_TWO_SCALE = 'CE(fused)+0.5*CE(detail)+0.5*CE(context), ignore 255'
SPECS = {
    'segformer_b0_shift': {'name': 'segformer_b0_shift', 'build': segformer_builder(SEGFORMER_B0), 'param_groups': single_group,
                           'lr': 6e-5, 'loss': _CE, 'pretrained': SEGFORMER_B0},
    'segformer_b2': {'name': 'segformer_b2', 'build': segformer_builder(SEGFORMER_B2), 'param_groups': single_group,
                     'lr': 6e-5, 'loss': _CE, 'pretrained': SEGFORMER_B2},
    'segformer_b2_hrda_c': {'name': 'segformer_b2_hrda_c', 'build': hrda_c_model, 'param_groups': single_group, 'lr': 6e-5,
                            'loss': _TWO_SCALE, 'pretrained': SEGFORMER_B2, 'dataset_factory': two_scale_dataset},
    'segformer_b2_hrda_d': {'name': 'segformer_b2_hrda_d', 'build': hrda_d_model, 'param_groups': hrda_d_groups, 'lr': 6e-5,
                            'loss': _TWO_SCALE + '; learned per-class scale attention (head lr x10)',
                            'pretrained': SEGFORMER_B2, 'dataset_factory': two_scale_dataset},
    'deeplabv3plus_resnet50': {'name': 'deeplabv3plus_resnet50', 'build': deeplab_model, 'param_groups': deeplab_groups,
                               'lr': 1e-4, 'loss': _CE, 'pretrained': RESNET50},
}
for _spec in SPECS.values():
    _spec['file'] = Path(__file__)
    _spec['description'] = __doc__
