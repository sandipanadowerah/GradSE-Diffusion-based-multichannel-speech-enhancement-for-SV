import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as trans
import soundfile as sf

def get_mel(x, featdim=40, sampling_rate=16000):
    
    instance_norm = nn.InstanceNorm1d(featdim)
    win_len = int(sampling_rate * 0.025)
    hop_len = int(sampling_rate * 0.01)
    
    feature_extract = trans.MelSpectrogram(sample_rate=sampling_rate, n_fft=512, win_length=win_len,
                                                        hop_length=hop_len, f_min=0.0, f_max=sampling_rate // 2,
                                                        pad=0, n_mels=featdim)
    
    for param in feature_extract.parameters():
        param.requires_grad = False
            
    
    with torch.no_grad():
        x = feature_extract(x) + 1e-6  # B x feat_dim x time_len
        x = x.log()
        x = instance_norm(x)    
    
    return x

def get_wav(filepath):
    data, sr = sf.read(filepath)
    return data

def get_feature(wavpath, nmics=3, sampling_rate=16000):
    x = []

    for i in range(nmics):
        x.append(get_wav(wavpath[:-5]+str(i+1)+'.wav'))
        
    x_np = np.asarray(x,dtype=np.float32)
    x_torch = torch.from_numpy(x_np)
    x_feat = get_mel(x_torch)
    
    return x_feat
