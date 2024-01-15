import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as trans

import torch.utils.data as data
from torch.utils.data import DataLoader
import soundfile as sf
import random

class SEDataset(torch.utils.data.Dataset):
    """
    dirpath: path to simulated data
    feattype: stft or mel
    sampling_rate: 
    nmics: number of channels
    mode: 1,2,3
    mode1: x: Noisy,   y: Target
    mode2: x: Noisy,   y: Dry target
    mode3: x: Target,  y: Dry target
    
    """
    def __init__(self, dirpath, feattype='stft', featdim=40, sampling_rate = 16000, nmics = 3, mode=1):
        self.dirpath = dirpath
        self.sampling_rate = sampling_rate
        self.mode = mode
        self.featdim = featdim
        self.feattype = feattype
        self.nmics = nmics
        self.fname_list = [fname.replace('.wav', '') for fname in os.listdir(os.path.join(dirpath, 'dry_target'))]
        
        if self.feattype == 'mel':
            
            self.instance_norm = nn.InstanceNorm1d(featdim)
            
            win_len = int(sampling_rate * 0.025)
            hop_len = int(sampling_rate * 0.01)
        
            self.feature_extract = trans.MelSpectrogram(sample_rate=sampling_rate, n_fft=512, win_length=win_len,
                                                        hop_length=hop_len, f_min=0.0, f_max=sampling_rate // 2,
                                                        pad=0, n_mels=featdim)
            
            for param in self.feature_extract.parameters():
                param.requires_grad = False
        
        if self.feattype == 'stft':
            self.nfft = self.featdim*2 - 1 
        
        random.seed(12345)
        random.shuffle(self.fname_list)
        
    def get_wav(self, filepath, multichannel=False):
        data, sr = sf.read(filepath)
        assert sr == self.sampling_rate
        return data
    
    def get_stft(self, x):
        return torch.stft(x, n_fft=self.nfft, return_complex=True)
    
    def get_mel(self, x):
        with torch.no_grad():
            x = self.feature_extract(x) + 1e-6  # B x feat_dim x time_len
            x = x.log()
            x = self.instance_norm(x)    
        return x
    
    def get_items(self, fname):
        x, y = [], []
        
        if self.mode == 1:
            
            for i in range(self.nmics):
                x.append(self.get_wav(os.path.join(self.dirpath, 'Noisy/'+fname+'_'+str(i+1)+'.wav')))
                
            y.append(self.get_wav(os.path.join(self.dirpath, 'Target/'+fname+'_'+str(1)+'.wav')))

        elif self.mode == 2:
            
            for i in range(self.nmics):
                x.append(self.get_wav(os.path.join(self.dirpath, 'Noisy/'+fname+'_'+str(i+1)+'.wav')))
                
            y.append(self.get_wav(os.path.join(self.dirpath, 'dry_target/'+fname+'.wav')))
            
        elif self.mode == 3:
            
            for i in range(self.nmics):
                x.append(self.get_wav(os.path.join(self.dirpath, 'Target/'+fname+'_'+str(i+1)+'.wav')))
                
            y.append(self.get_wav(os.path.join(self.dirpath, 'dry_target/'+fname+'.wav')))
            
        
        x_np = np.asarray(x,dtype=np.float32)
        y_np = np.asarray(y,dtype=np.float32)
        
        x_torch = torch.from_numpy(x_np)
        y_torch = torch.from_numpy(y_np)
        
        min_len = min(x_torch.shape[-1], y_torch.shape[-1])
        x_torch = x_torch[:,:min_len]
        y_torch = y_torch[:,:min_len]
        
        if self.feattype == 'stft':
            #print(x_torch.shape, y_torch.shape)
            x_feat = self.get_stft(x_torch)
            y_feat = self.get_stft(y_torch)
            #print(x_feat.shape, y_feat.shape)
            
        elif self.feattype == 'mel':
            x_feat = self.get_mel(x_torch)
            y_feat = self.get_mel(y_torch)
            
        else:
            print('Unknown feattype!!!')
        
        
        return x_feat, y_feat

    def __getitem__(self, index):
        x,y = self.get_items(self.fname_list[index])
        item = {'src': x, 'trg': y}
        return item

    def __len__(self):
        return len(self.fname_list)


class SEBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        
        src_max_length = max([item['src'].shape[-1] for item in batch])        
        trg_max_length = max([item['trg'].shape[-1] for item in batch])
        assert src_max_length == trg_max_length
        
        nmic_src = batch[0]['src'].shape[0]
        featdim_src = batch[0]['src'].shape[1]
        
        nmic_trg = batch[0]['trg'].shape[0]
        featdim_trg = batch[0]['trg'].shape[1]
        
        src = torch.zeros((B, nmic_src, featdim_src, src_max_length), dtype=torch.float32)
        trg = torch.zeros((B, nmic_trg, featdim_trg, trg_max_length), dtype=torch.float32)
        
        for i, item in enumerate(batch):
            src_, trg_ = item['src'], item['trg']            
            src[i,:,:,:src_.shape[-1]] = src_
            trg[i,:,:,:trg_.shape[-1]] = trg_
            
        return {'x': src, 'y': trg}
    
    
# batch_size x nmic x featdim x frameLen

#dirpath = '/home/ajinkyak/Videos/dereverb/sample_dataset'

#train_dataset = SEDataset(dirpath, feattype='stft', featdim=256, mode=3)

#batch_collate = SEBatchCollate()

#train_loader = DataLoader(dataset=train_dataset, batch_size=1,
#                            collate_fn=batch_collate, drop_last=True,
#                            num_workers=4, shuffle=False)

#for batch_idx, batch in enumerate(train_loader):
#    x,y  = batch['src'], batch['trg']
#
#    print(x.shape, y.shape)

# batch_size x nmic x featdim x frameLen

#dirpath = '/home/ajinkyak/Videos/dereverb/sample_dataset'

#train_dataset = SEDataset(dirpath, feattype='mel', featdim=40, mode=3)

#batch_collate = SEBatchCollate()

#train_loader = DataLoader(dataset=train_dataset, batch_size=1,
#                            collate_fn=batch_collate, drop_last=True,
#                            num_workers=4, shuffle=False)

#for batch_idx, batch in enumerate(train_loader):
#    x,y  = batch['src'], batch['trg']

#    print(x.shape, y.shape)
