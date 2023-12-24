import torch
import torch.nn as nn
import torch.nn.functional as F

class NLayerEncoder(nn.Module):

    def __init__(self, input_nc=3, ndf=64, n_layers=3):

        super().__init__()
        norm_layer = nn.BatchNorm2d

        kw = 3
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=5, stride=2, padding=2), nn.ReLU()]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.ReLU()
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=1, padding=1),
            norm_layer(ndf * nf_mult),
            nn.ReLU()
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=1, stride=1, padding=0),]

        self.main = nn.Sequential(*sequence)

    def forward(self, input):

        return self.main(input)