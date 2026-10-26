import torch
import torch.nn as nn
import torchvision.models as models


class MRNetPlane(nn.Module):
    """One plane model: AlexNet features per slice, global average pool, max over slices, one logit."""

    def __init__(self):
        super().__init__()
        alex = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.features = alex.features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)                       # grayscale to 3 channels for ImageNet weights
        f = self.gap(self.features(x)).flatten(1)      # (n_slices, 256)
        v = f.max(dim=0).values                        # (256,) exam-level vector
        return self.fc(v)                              # (1,) logit



class ReportLabelModel:
    """MedGemma model for extracting findings from knee MRI reports."""

    def __init__(self, model_path, use_4bit=True):
        from transformers import (
            AutoProcessor,
            AutoModelForImageTextToText,
            BitsAndBytesConfig,
        )

        self.processor = AutoProcessor.from_pretrained(model_path)

        if use_4bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                quantization_config=bnb,
                device_map={"": 0},
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map={"": 0},
            )

        self.model.eval()
        print("GPU memory (GB):", torch.cuda.memory_allocated() / 1e9)