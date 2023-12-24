import torch
import torch.nn as nn
import torch.nn.functional as F

from swin_transformer import *
from decoder import *
from discriminator import NLayerDiscriminator



class BaseModel(nn.Module):
    def __init__(self, img_size=512, n_embedding=4096, dim=512):
        super().__init__()

        self.encoder = Swin_encoder(img_size=img_size)
        self.codebook = nn.Parameter(torch.randn(n_embedding, dim))
        self.decoder = DeformableDecoder(img_size=img_size, n_embedding=n_embedding, dim=dim)

    def foward(self, x):
        

    