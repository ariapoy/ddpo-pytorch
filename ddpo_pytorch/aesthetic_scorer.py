# Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py

from importlib import resources
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

ASSETS_PATH = resources.files("ddpo_pytorch.assets")

prefix_path = '/tmp2/model_assets'

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(f"{prefix_path}/openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained(f"{prefix_path}/openai/clip-vit-large-patch14")
        self.mlp = MLP()
        state_dict = torch.load(
            ASSETS_PATH.joinpath(f"{prefix_path}/sac+logos+ava1-l14-linearMSE.pth")
        )
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)

class CLIPScore(torch.nn.Module):
    def __init__(self, dtype):
        self.clip = CLIPModel.from_pretrained(f"{prefix_path}/openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained(f"{prefix_path}/openai/clip-vit-large-patch14")
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images, prompts):
        device = next(self.parameters()).device
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        outputs = self.clip(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        return logits_per_image

if __name__ == '__main__':
    pass