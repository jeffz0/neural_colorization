import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import collections

import numpy as np
import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms

class DeterministicEncoder(nn.Module):
    def __init__(self, output_sizes):
        '''
        CNP encoder

        @param output_sizes: An iterable containing the output sizes of the encoding MLP.
        '''
        super(DeterministicEncoder, self).__init__()
        encoder = [nn.Linear(42, output_sizes[0]), nn.ReLU(inplace=True)]
        for i in range(1,len(output_sizes)):
            encoder += [nn.Linear(output_sizes[i - 1], output_sizes[i])]
            if i != len(output_sizes) - 1:
                encoder += [nn.ReLU(inplace=True)]

        self.encoder = nn.Sequential(*encoder)


    def forward(self, context):
        '''
        Encodes the inputs into one representation.

        @param context_x: Tensor of size bs x observations x m_ch. For this 1D regression
            task this corresponds to the x-values.
        @param context_y: Tensor of size bs x observations x d_ch. For this 1D regression
            task this corresponds to the y-values.
        @param num_context_points: A tensor containing a single scalar that indicates the
            number of context_points provided in this iteration.
        @return representation: The encoded representation averaged over all context 
            points.
        '''
        
        encoder_input = context
        batch_size, num_context_points, _ = encoder_input.shape
        encoder_input = encoder_input.view(batch_size * num_context_points, -1)
        representation = self.encoder(encoder_input).view(batch_size, num_context_points, -1)
        return torch.mean(representation, dim=1)
    
class DeterministicDecoder(nn.Module):
    def __init__(self, output_sizes):
        '''
        CNP decoder

        @param output_sizes: An iterable containing the output sizes of the decoder MLP 
          as defined in `nn.Linear`.
        '''
        super(DeterministicDecoder, self).__init__()
        decoder = [nn.Linear(512+40, output_sizes[0]), nn.ReLU(inplace=True)]
        for i in range(1, len(output_sizes)):
            decoder += [nn.Linear(output_sizes[i - 1], output_sizes[i])]
            if i != len(output_sizes) - 1:
                decoder += [nn.ReLU(inplace=True)]

        self.decoder = nn.Sequential(*decoder)


    def forward(self, decoder_input):
        '''
        Decodes the individual targets.

        @param representation: The encoded representation of the context
        @param target_x: The x locations for the target query
        @param num_total_points: The number of target points.
        @return dist: A multivariate Gaussian over the target points.
        @return mu: The mean of the multivariate Gaussian.
        @return sigma: The standard deviation of the multivariate Gaussian.
        '''
        batch_size, num_total_points, _ = decoder_input.shape
        decoder_input = decoder_input.view(batch_size * num_total_points, -1)

        output = self.decoder(decoder_input).view(batch_size, num_total_points, -1)

        # get the mean and variance
        mu_a, log_sigma_a, mu_b, log_sigma_b = torch.chunk(output, 4, dim=-1)

        # bound the variance
        sigma_a = 0.1 + 0.9 * F.softplus(log_sigma_a)
        sigma_b = 0.1 + 0.9 * F.softplus(log_sigma_b)

        # get the distribution
        dist_a = torch.distributions.multivariate_normal.MultivariateNormal(mu_a, covariance_matrix=torch.diag_embed(sigma_a))
        dist_b = torch.distributions.multivariate_normal.MultivariateNormal(mu_b, covariance_matrix=torch.diag_embed(sigma_b))

        return dist_a,  mu_a,  log_sigma_a, dist_b, mu_b, log_sigma_b
    
    
def get_encoder(output_sizes):
    return DeterministicEncoder(output_sizes)

def get_decoder(output_sizes):
    return DeterministicDecoder(output_sizes)