# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params
from model import GradDeReverb
from dataset import SEDataset, SEBatchCollate
from utils import plot_tensor, save_plot


train_filelist_path = params.train_dirpath_path
valid_filelist_path = params.valid_dirpath_path

feattype = params.feattype
mode = params.mode
n_feats = params.n_feats

log_dir = params.log_dir
n_epochs = params.n_epochs
batch_size = params.batch_size
out_size = params.out_size

learning_rate = params.learning_rate
random_seed = params.seed

dec_dim = params.dec_dim
beta_min = params.beta_min
beta_max = params.beta_max
pe_scale = params.pe_scale

outdir = params.outdir

checkpoint_epoch = params.checkpoint_epoch

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == "__main__":
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print('Initializing logger...')
    logger = SummaryWriter(log_dir=log_dir)

    print('Initializing data loaders...')
    #train_dataset = SEDataset(train_filelist_path, feattype=feattype, featdim=n_feats, sampling_rate = 16000, nmics = 3, mode=mode)
    batch_collate = SEBatchCollate()
    
    test_dataset = SEDataset(valid_filelist_path, feattype=feattype, featdim=n_feats, sampling_rate = 16000, nmics = 3, mode=mode)

    print('Initializing model...')
    model = GradDeReverb(n_feats, dec_dim, beta_min, beta_max, pe_scale)
    
    print('Number of decoder parameters: %.2fm' % (model.decoder.nparams/1e6))
    print('Total parameters: %.2fm' % (model.nparams/1e6))    
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print('Initializing optimizer...')
    
    
    if checkpoint_epoch != None:
        print('Loading checkpoint from ', checkpoint_epoch)    
        ckpt = torch.load(f"{log_dir}/grad_{checkpoint_epoch}.pt")
        model.load_state_dict(ckpt['model'])
        model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=ckpt['lr'])
        optimizer.load_state_dict(ckpt['optimizer'])
        n_start = checkpoint_epoch
    else:
        model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        n_start = 1
        
    
    model.eval()
    with torch.no_grad():
        for i, item in enumerate(test_dataset):
            x = item['src'].unsqueeze(0).cuda().float()
            y = item['trg'].unsqueeze(0).cuda().squeeze(1).float()
            len_ = int((x.shape[-1]//8)*8)
            x = x[:,:,:,:len_]
            #g2 = item['y'].unsqueeze(0).cuda().permute(0,2,1)
                
            print(x.shape, y.shape)
            y_enc = model(x, n_timesteps=1000)
            out_data = {'y_orig': y.squeeze()[:,:len_], 'y_out': y_enc.squeeze()[:,:len_]}
            torch.save(out_data, os.path.join(outdir, f'output_{i}.pt'))
                
            save_plot(x.squeeze()[0].cpu(), 
                          f'input_{i}.png')
            save_plot(y_enc.squeeze().cpu(), 
                          f'generated_enc_{i}.png')
            save_plot(y.squeeze().cpu(), 
                          f'orignal_{i}.png')
                
        
