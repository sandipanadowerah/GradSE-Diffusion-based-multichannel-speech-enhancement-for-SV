# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
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
    train_dataset = SEDataset(train_filelist_path, feattype=feattype, featdim=n_feats, sampling_rate = 16000, nmics = 3, mode=mode)
    batch_collate = SEBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=4, shuffle=False)
    test_dataset = SEDataset(valid_filelist_path, feattype=feattype, featdim=n_feats, sampling_rate = 16000, nmics = 3, mode=mode)

    print('Initializing model...')
    model = GradDeReverb(n_feats, dec_dim, beta_min, beta_max, pe_scale)
    #print('Number of encoder : %.2fm' % (model.encoder.nparams/1e6))
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
        
    
    print('Logging test batch...')

    print('Start training...')
    iteration = 0
    for epoch in range(n_start, n_epochs + 1):
        model.train()
        prior_losses = []
        diff_losses = []
        with tqdm(loader, total=len(train_dataset)//batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                x = batch['x'].cuda()#.squeeze(1)
                y = batch['y'].cuda().squeeze(1)
                
                if out_size != None:
                    x = x[:,:,:,:out_size]
                    y = y[:,:,:out_size]
                                   
                
                #print(x.shape, y.shape)
                
                prior_loss, diff_loss = model.compute_loss(x,y)
                loss = sum([prior_loss, diff_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),
                                                               max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),
                                                               max_norm=1)
                optimizer.step()

                
                logger.add_scalar('training/prior_loss', prior_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/diffusion_loss', diff_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                  global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                  global_step=iteration)
                
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                
                #print(epoch)
                
                if batch_idx % 5 == 0:
                    msg = f'Epoch: {epoch}, iteration: {iteration} | prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}'
                    progress_bar.set_description(msg)
                
                iteration += 1

        log_msg = '| prior loss = %.3f ' % np.mean(prior_losses)
        log_msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
        with open(f'{log_dir}/train.log', 'a') as f:
            f.write(log_msg)

        if epoch % params.save_every > 0:
            continue

        model.eval()
        with torch.no_grad():
            for i, item in enumerate(test_dataset):
                if i < 3:
                    x = item['src'].unsqueeze(0).cuda().float()
                    y = item['trg'].unsqueeze(0).cuda().squeeze(1).float()
                    #g2 = item['y'].unsqueeze(0).cuda().permute(0,2,1)
                    len_ = int((x.shape[-1]//8)*8)
                    x = x[:,:,:,:len_]
                    #print(x.shape, 'x')
                    y_enc = model(x, n_timesteps=50)
                    #print(y_enc.shape, 'y_enc')
                    logger.add_image(f'image_{i}/generated_enc',
                                     plot_tensor(y_enc.squeeze().cpu()),
                                     global_step=iteration, dataformats='HWC')

                    logger.add_image(f'image_{i}/orignal_dec',
                                     plot_tensor(y.squeeze().cpu()),
                                     global_step=iteration, dataformats='HWC')
                    save_plot(y_enc.squeeze().cpu(), 
                              f'{log_dir}/generated_enc_{i}.png')
                    save_plot(y.squeeze().cpu(), 
                              f'{log_dir}/generated_dec_{i}.png')

        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 'lr': get_lr(optimizer)}, f=f"{log_dir}/grad_{epoch}.pt")
