import torch
import torch.nn as nn
import torch.nn.functional as F

from swin_transformer import *
from decoder import *
from discriminator import NLayerDiscriminator
from timm.models.layers import to_2tuple



class BaseModel(nn.Module):
    def __init__(self, img_size=512, num_queries=4096, 
                 n_embedding=4096, dim=512,
                 dim_feedforward=1024, nhead=8,dropout=0.1,
                 num_decoder_layers=6, activation="relu", 
                 num_feature_levels=1, dec_n_points=4,
                 view_dim=10):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.dim = dim

        # self.encoder = Swin_encoder(img_size=img_size)
        self.codebook = nn.Embedding(n_embedding, dim)
        self.codebook.weight.data.uniform_(-1.0 / dim, 1.0 / dim)

        self.queries = nn.Embedding(num_queries, dim*2)
        decoder_layer = DeformableTransformerDecoderLayer(dim, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers)
        self.reference_points = nn.Linear(dim, 2)
        self.view_proj = nn.Linear(view_dim, dim*2)


    def foward(self, view, feature=None):
        
        if feature is not None:
            N, C, H, W = feature.shape
            ze = feature.permute(0, 2, 3, 1).contiguous().view(-1, C)

            distance = torch.sum(ze ** 2, dim=1, keepdim=True) + \
                torch.sum(self.codebook.weight**2, dim=1) - 2 * \
                torch.matmul(ze, self.codebook.weight.t())
            
            min_encoding_indices = torch.argmin(distance, dim=1)
            zq = self.codebook(min_encoding_indices)

            decoder_input = ze + (zq - ze).detach()

        else:
            decoder_input = self.codebook.weight.data
        

        view_embed = self.view_proj(view)
        query_embed = self.queries.weight + view_embed
        query_embed, tgt = torch.split(query_embed, C, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(N, -1, -1)
        tgt = tgt.unsqueeze(0).expand(N, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        

    