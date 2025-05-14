"""SegFormer-B0 with a single-channel anomaly head."""

import torch.nn as nn
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor


def load_model():
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    model = AutoModelForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        id2label={0: "background", 1: "anomaly"},
        ignore_mismatched_sizes=True,  # allows us to swap heads
    )
    hid = model.config.hidden_size
    model.decode_head = nn.Conv2d(hid, 1, kernel_size=1)  # 1 channel
    return processor, model
