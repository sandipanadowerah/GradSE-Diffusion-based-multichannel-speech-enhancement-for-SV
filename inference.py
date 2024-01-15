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

import params_eval as params
import eval_utils 
from model import GradDeReverb
from utils import plot_tensor, save_plot


test_filelist_path = params.test_filelist_path
test_dirpath = params.test_dirpath

feattype = params.feattype
mode = params.mode
n_feats = params.n_feats

log_dir = params.log_dir

dec_dim = params.dec_dim
beta_min = params.beta_min
beta_max = params.beta_max
pe_scale = params.pe_scale

outdir = params.outdir

checkpoint_epoch = params.checkpoint_epoch

if __name__ == "__main__":

    print('Initializing data loaders...')
    
    #test_dataset = SEDataset(valid_filelist_path, feattype=feattype, featdim=n_feats, sampling_rate = 16000, nmics = 3, mode=mode)

    print('Initializing model...')
    model = GradDeReverb(n_feats, dec_dim, beta_min, beta_max, pe_scale)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
    
    
    if checkpoint_epoch != None:
        print('Loading checkpoint from ', checkpoint_epoch)    
        ckpt = torch.load(f"{log_dir}/grad_{checkpoint_epoch}.pt")
        model.load_state_dict(ckpt['model'])
        model.to(device)
    else:
        print('Error checkpoint_epoch not provided')
        
    
    test_data = [os.path.join(test_dirpath,fname.strip()) for fname in open(test_filelist_path, 'r')]
    
    model.eval()
    
    with torch.no_grad():
        for i, fpath in enumerate(test_data):
        
            x = eval_utils.get_feature(fpath)
            x = x.unsqueeze(0).cuda().float()
            
            len_ = int((x.shape[-1]//8)*8)
            x = x[:,:,:,:len_]
            #g2 = item['y'].unsqueeze(0).cuda().permute(0,2,1)
                
            print(x.shape)
            y_enc = model(x, n_timesteps=500)
            out_data = {'x': x.cpu().squeeze()[:,:len_], 'y_out': y_enc.cpu().squeeze()[:,:len_]}
            torch.save(out_data, os.path.join(outdir, f'output_{i}.pt'))
                
            save_plot(x.squeeze()[0].cpu(), 
                          f'{outdir}/input_{i}.png')
            save_plot(y_enc.squeeze().cpu(), 
                          f'{outdir}/generated_enc_{i}.png')
            
                
        
