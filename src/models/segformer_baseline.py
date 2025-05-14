"""SegFormer-B0 with a 1-channel anomaly head."""

import torch.nn as nn
from transformers import (
    AutoModelForSemanticSegmentation,
    SegformerImageProcessor,
)


def load_model():
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )

    model = AutoModelForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        id2label={0: "anomaly"},  # ONE class
        ignore_mismatched_sizes=True,
    )

    in_ch = model.decode_head.classifier.in_channels  # 256 for B0
    model.decode_head.classifier = nn.Conv2d(in_ch, 1, 1)
    model.config.num_labels = 1
    model.config.label2id = {"anomaly": 0}

    return processor, model
