# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
import torch


class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def nparams(self):
        """
        Returns number of trainable parameters of the module.
        """
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params += np.prod(param.detach().cpu().numpy().shape)
        return num_params


    def relocate_input(self, x: list):
        """
        Relocates provided tensors to the same device set for the module.
        """
        device = next(self.parameters()).device
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x
    
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(CRNN, self).__init__()
        
        self.conv = nn.Conv2d(3,1,3,1,1)
        self.rnn = nn.LSTM(feat_dim, hidden_dim, 4, batch_first=True)
        
    def forward(self, x):
        #print(x.shape, 'input to crnn')    
        #x = x.permute(0,1,3,2)
        x = x.permute(0,1,3,2)
        #x = x.unsqueeze(1)
        #print(x.shape, 'before conv crnn')
        o = self.conv(x)
        #print(o.shape, 'after conv')
        o = o.squeeze(1)
        #print(o.shape, 'before rnn')
        o,_ = self.rnn(o)
        #print(o.shape, 'after rnn')
        o = o.permute(0,2,1)
        #print(o.shape, 'after permute 0,2,1')
        x_mask = torch.ones(o.shape[0], 1, o.shape[2]).to(o.dtype).cuda()
        #print(x_mask.shape, 'x_mask')
        return o, x_mask # batch_size x seq_len x feat_dim, 
    
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import torch
from einops import rearrange

class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(BaseModule):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(BaseModule):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Rezero(BaseModule):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim_out, 3, 
                                         padding=1), torch.nn.GroupNorm(
                                         groups, dim_out), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(BaseModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                               dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(BaseModule):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)            

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
                            heads = self.heads, qkv=3)            
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', 
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Residual(BaseModule):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(BaseModule):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GradLogPEstimator2d(BaseModule):
    def __init__(self, dim, dim_mults=(1, 2, 4), groups=8, pe_scale=1000):
        super(GradLogPEstimator2d, self).__init__()
        self.pe_scale = pe_scale
        
        dims = [2, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim * 4), Mish(),
                                       torch.nn.Linear(dim * 4, dim))

        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                       ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                       ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                       Residual(Rezero(LinearAttention(dim_out))),
                       Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                     ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                     ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                     Residual(Rezero(LinearAttention(dim_in))),
                     Upsample(dim_in)]))
        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self, x, mask, mu, t):
        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        x = torch.stack([mu, x], 1)
        mask = mask.unsqueeze(1)

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)


def get_noise(t, beta_init, beta_term, cumulative=False):
    if cumulative:
        noise = beta_init*t + 0.5*(beta_term - beta_init)*(t**2)
    else:
        noise = beta_init + (beta_term - beta_init)*t
    return noise


class Diffusion(BaseModule):
    def __init__(self, n_feats, dim, beta_min, beta_max, pe_scale):
        super(Diffusion, self).__init__()
        self.estimator = GradLogPEstimator2d(dim, pe_scale=pe_scale)
        self.n_feats = n_feats
        self.dim = dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

    def forward_diffusion(self, x0, mask, mu, t):
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        mean = x0*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
                                                 device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max, 
                                cumulative=False)
            if stoc:  # adds stochastic term
                dxt_det = 0.5 * (mu - xt) - self.estimator(xt, mask, mu, t)
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - self.estimator(xt, mask, mu, t))
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def forward(self, z, mask, mu, n_timesteps, stoc=False):
        return self.reverse_diffusion(z, mask, mu, n_timesteps, stoc)

    def loss_t(self, x0, mask, mu, t):
        #print(x0.shape, mask.shape, mu.shape, t.shape)
        xt, z = self.forward_diffusion(x0, mask, mu, t)
        #print(xt.shape, z.shape)
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        
        #print(xt.shape, mask.shape, mu.shape, t.shape)
        
        noise_estimation = self.estimator(xt, mask, mu, t)
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
        
        #print(noise_estimation.shape, z.shape, mask.shape, self.n_feats)
        
        loss =  torch.sum((noise_estimation + z)**2) / (torch.sum(mask)*self.n_feats)
        
        return loss

    def compute_loss(self, x0, mask, mu, offset=1e-5):
        
        #print(x0.shape, x0.shape[0], x0.dtype, x0.device)
        
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device, 
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        
        #print(x0.shape, mask.shape, mu.shape, t.shape)
        
        return self.loss_t(x0, mask, mu, t)
        
        
        
class GradDeReverb(BaseModule):
    def __init__(self, n_feats, dec_dim, beta_min, beta_max, pe_scale):
        super(GradDeReverb, self).__init__()
        
        
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        self.encoder = CRNN(n_feats, n_feats)
        
        self.decoder = Diffusion(n_feats, dec_dim, beta_min, beta_max, pe_scale)

    @torch.no_grad()
    def forward(self, x, n_timesteps, temperature=1.0):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        #print(x.shape)
        
        x = self.relocate_input(x)
        
        #print(x.shape)
        
        mu_y, y_mask = self.encoder(x) # todo

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature        
        
        # Generate sample by performing reverse dynamics
        #print(z.shape, y_mask.shape, mu_y.shape, n_timesteps)
        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps)

        return decoder_outputs

    def compute_loss(self, x, y, out_size=None):
        """
        Computes 2 losses:
            1. prior loss: loss between mel-spectrogram and encoder outputs.
            2. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            x (torch.Tensor): batch of noisy/reverb mel-spectrograms/spectrograms.
            x_lengths (torch.Tensor): lengths of input in batch.
            
            y (torch.Tensor): batch of corresponding mel-spectrograms/spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms/spectrograms in batch.
            
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        x, y = self.relocate_input([x, y])
        
        mu_y, y_mask = self.encoder(x)
        
        #print(y.shape, y_mask.shape, mu_y.shape)
        
        # Compute loss of score-based decoder
        diff_loss = self.decoder.compute_loss(y, y_mask, mu_y)
        
        # Compute loss between aligned encoder outputs and mel-spectrogram
        #prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        
        prior_loss = nn.MSELoss()(mu_y, y)
        
        
        
        #prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        
        return prior_loss, diff_loss
    
    
#dereverb_model = GradDeReverb(n_feats=40, dec_dim=64, beta_min=0.05, beta_max=20.0, pe_scale=1000)

#x = torch.rand(2,40,100)
#y = torch.rand(2,40,100)
#n_timesteps=10
#temperature=1.0

#loss = dereverb_model.compute_loss(x,y)
#dec_out = dereverb_model(x,n_timesteps=10)

#loss, dec_out.shape
