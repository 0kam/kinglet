# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Student backbones and the log-mel settings shared by training, scoring and export.

A backbone maps a normalised log-mel (B, 1, 128, 501) of a 5-s window at 32 kHz to a feature vector
(B, num_features). Names are timm model names: regnety_016 (B), efficientnet_lite0 (S) and
mobilenetv4_conv_small (XS) in the paper.
"""

import torch

SR = 32_000
MEL = {"n_fft": 1024, "hop_length": 320, "n_mels": 128}  # (128, 501) per 5 s window


def build(name: str) -> torch.nn.Module:
    import timm

    return timm.create_model(name, pretrained=False, num_classes=0, in_chans=1)
