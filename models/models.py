import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

import numpy as np

from .helphers import parse_features_numbers


class ReshapeLayer(nn.Module):
    def __init__(self, target_shape):
        super(ReshapeLayer, self).__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.reshape(self.target_shape)


class EncoderBlock(nn.Module):
    def __init__(self, inp_features, out_features):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(inp_features, out_features, (3, 3), padding=1)
        self.bn = nn.BatchNorm2d(out_features)
        self.mp = nn.MaxPool2d((2,2))
        self.non_lin = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.non_lin(self.mp(self.bn(self.conv(x))))

    def output_shape(self, inp_shape):
        return (inp_shape // 2)


class DecoderBlock(nn.Module):
    def __init__(self, inp_features, out_features, non_lin=nn.LeakyReLU(0.2)):
        super(DecoderBlock, self).__init__()
        self.ump = nn.ConvTranspose2d(inp_features, inp_features, (2,2), stride=2)
        self.tconv = nn.Conv2d(inp_features, out_features, (3, 3), padding=1)
        self.bn = nn.BatchNorm2d(out_features)
        self.non_lin = non_lin
        #self.bn = lambda x:x

    def forward(self, x):
        return self.non_lin(self.bn(self.tconv(self.ump(x))))

    def output_shape(self, inp_shape):
        return (inp_shape * 2)


class Encoder(nn.Module):
    def __init__(self, inp_shape, internal_shape, feature_maps=(1,16,32,64)):
        super(Encoder, self).__init__()

        self.internal_shape = internal_shape

        self.blocks = nn.Sequential(*[EncoderBlock(*x) for x in parse_features_numbers(feature_maps)])

        self.last_conv_shape = inp_shape
        for b in self.blocks:
            print(self.last_conv_shape)
            self.last_conv_shape = b.output_shape(self.last_conv_shape)

        self.fc_shape = self.last_conv_shape[0] * self.last_conv_shape[1] * 64

        self.fc_mu = nn.Linear(self.fc_shape, self.internal_shape)
        self.fc_sigma = nn.Linear(self.fc_shape, self.internal_shape)


    def forward(self, x):
        x = self.blocks(x)

        return self.fc_mu(x.reshape(-1, self.fc_shape)),\
               self.fc_sigma(x.reshape(-1, self.fc_shape))



class Decoder(nn.Module):
    def __init__(self, fc_shape, internal_shape, feature_maps=(1,16,32,64)):
        super(Decoder, self).__init__()

        self.internal_shape = internal_shape
        self.fc_shape = fc_shape

        self.fc = nn.Linear(internal_shape, 64 * fc_shape[0] * fc_shape[1])
        self.reshape_layer = ReshapeLayer([-1, 64, self.fc_shape[0], self.fc_shape[1]])
        # self.blocks = nn.Sequential(DecoderBlock(64, 32),
        #                             DecoderBlock(32, 16),
        #                             DecoderBlock(16, 1, nn.Sigmoid()))

        feature_maps = parse_features_numbers(feature_maps[::-1])
        layers = [DecoderBlock(*x) for x in feature_maps[:-1]]
        layers.append(DecoderBlock(feature_maps[-1][0], feature_maps[-1][1], nn.Sigmoid()))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        x = self.reshape_layer(x)
        x = nn.LeakyReLU(0.2)(x)

        return self.blocks(x)


class VAE(nn.Module):
    def __init__(self, input_shape, internal_shape, feature_map=(1,16,32,64)):
        super(VAE, self).__init__()

        self.input_shape = np.array(list(input_shape))
        self.internal_shape = internal_shape
        self.feature_map = feature_map

        self.enc = Encoder(self.input_shape, self.internal_shape, self.feature_map)
        self.dec = Decoder(self.enc.last_conv_shape, self.internal_shape, self.feature_map)

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparametrize(mu, logvar)
        restored_x = self.dec(z)
        return restored_x, mu, logvar

    def reparametrize(self, mu, logvar):
        z = torch.randn_like(mu)
        return z.mul(torch.exp(0.5 * logvar)).add_(mu)

