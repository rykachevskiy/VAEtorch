import torch
import torch.utils.data
from torch import nn

import numpy as np


class ReshapeLayer(nn.Module):
    def __init__(self, target_shape):
        super(ReshapeLayer, self).__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.reshape(self.target_shape)


class EncoderBlock(nn.Module):
    def __init__(self, inp_features, out_features):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(inp_features, out_features, (3, 3))#, padding=1)
        self.mp = nn.MaxPool2d((2, 2), return_indices=True)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x, mp_idx = self.mp(self.conv(x))
        return self.bn(x), mp_idx

    def output_shape(self, inp_shape):
        return (inp_shape - 2) // 2


class DecoderBlock(nn.Module):
    def __init__(self, inp_features, out_features):
        super(DecoderBlock, self).__init__()
        self.upmp = nn.MaxUnpool2d((2, 2))
        self.upconv = nn.ConvTranspose2d(inp_features, out_features, (3, 3), padding=1)

        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x, mp_idx = self.mp(self.conv(x))
        return self.bn(x), mp_idx

    def output_shape(self, inp_shape):
        return (inp_shape - 2) // 2


class Encoder(nn.Module):
    def __init__(self, inp_shape, internal_shape):
        super(Encoder, self).__init__()

        self.internal_shape = internal_shape

        self.blocks = [EncoderBlock(1, 8),
                       EncoderBlock(8, 16),
                       EncoderBlock(16, 32)]

        self.fc_shape = inp_shape
        for b in self.blocks:
            self.fc_shape = b.output_shape(self.fc_shape)

        print(self.fc_shape)
        self.fc_shape = self.fc_shape[0] * self.fc_shape[1] * 32

        self.fc_mu = nn.Linear(self.fc_shape, self.internal_shape)
        self.fc_sigma = nn.Linear(self.fc_shape, self.internal_shape)

    def forward(self, x):
        mp_idxs = []
        for block in self.blocks:
            x, mp_idx = block(x)
            x = nn.LeakyReLU(0.2)(x)
            mp_idxs.append(mp_idx)

        return self.fc_mu(x.reshape(-1, self.fc_shape)),\
               self.fc_sigma(x.reshape(-1, self.fc_shape)),\
               mp_idx



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.after_conv_shape = 7*7*16
        self.internal_shape = 10
        self.up_convs = nn.Sequential(nn.Linear(self.internal_shape, self.after_conv_shape),
                                      nn.BatchNorm1d(self.after_conv_shape),
                                      nn.LeakyReLU(0.1),
                                      ReshapeLayer((-1, 16, 7, 7)),

                                      nn.MaxUnpool2d((2,2)),
                                      nn.Conv2d(16, 8, (3, 3), padding=1),
                                      nn.LeakyReLU(0.1),
                                      nn.BatchNorm2d(8),

                                      nn.Upsample((28, 28)),
                                      nn.Conv2d(8, 1, (3, 3), padding=1),
                                      nn.Sigmoid())

    def simple_forward(self, x):
        return self.up_convs(x)

    def forward(self, mu, log_sigma):
        return self.simple_forward(self.reparametrize(mu, log_sigma))

    def sample_hidden(self, batch_size=1):
        z = torch.randn((batch_size, self.internal_shape))
        return z

    def reparametrize(self, mu, log_sigma):
        batch_size=mu.shape[0]
        return mu + self.sample_hidden(batch_size) * torch.exp(0.5*log_sigma) #std = torch.exp(0.5 * logvar)

# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#
#         self.after_conv_shape = 7*7*16
#         self.internal_shape = 100
#         self.up_convs = nn.Sequential(nn.Linear(self.internal_shape, self.after_conv_shape),
#                                       nn.BatchNorm1d(self.after_conv_shape),
#                                       nn.LeakyReLU(0.1),
#                                       ReshapeLayer((-1, 16, 7, 7)),
#
#                                       nn.Upsample((14,14)),
#                                       nn.Conv2d(16, 8, (3, 3), padding=1),
#                                       nn.LeakyReLU(0.1),
#                                       nn.BatchNorm2d(8),
#
#                                       nn.Upsample((28, 28)),
#                                       nn.Conv2d(8, 1, (3, 3), padding=1),
#                                       nn.Sigmoid())
#
#     def simple_forward(self, x):
#         return self.up_convs(x)
#
#     def forward(self, mu, log_sigma):
#         return self.simple_forward(self.reparametrize(mu, log_sigma))
#
#     def sample_hidden(self, batch_size=1):
#         z = torch.randn((batch_size, self.internal_shape))
#         return z
#
#     def reparametrize(self, mu, log_sigma):
#         batch_size=mu.shape[0]
#         return mu + self.sample_hidden(batch_size) * torch.exp(0.5*log_sigma) #std = torch.exp(0.5 * logvar)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.enc = Encoder()
        self.dec = Decoder()

    def forward(self, x):
        pass