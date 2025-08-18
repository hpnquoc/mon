# This is how we compute the empty embedding
# You may need to download 'openai/clip-vit-large-patch14'

import pickle

import albumentations as A
import box
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import mon
from mon import console, metrics, Path, tfms, optims
from ldm.modules.encoders.modules import FrozenCLIPEmbedder

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]

model         = FrozenCLIPEmbedder().to("cuda")
embedding     = model.encode([""]).cpu()

print(embedding)
print(embedding.shape)

with open(str(current_dir / "empty_embedding.pkl"), "wb") as f:
    pickle.dump(embedding, f)
