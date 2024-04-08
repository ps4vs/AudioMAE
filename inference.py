import torch
from torchvision import transforms
from datasets import load_dataset, Dataset, Audio
from torchaudio.compliance import kaldi

import os
import librosa
import requests
import importlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import models_mae

MELBINS = 128
TARGET_LEN =1024
DATASET = load_dataset("agkphysics/audioset", split="test", streaming=True)

def prepare_model(ckpt_dir='./ckpt/pretrained.pth', arch='mae_vit_small_patch16'):
    ## LOAD MODEL
    model = getattr(models_mae, arch)(in_chans=1, audio_exp=True, img_size=(1024, 128), decoder_mode=0)
    # this checkpoint dir is being loaded directly
    checkpoint = torch.load(ckpt_dir, map_location='cpu')
    ## EQUIQ WEIGHTS
    model.load_state_dict(checkpoint[arch], strict=False)
    # when the model weights are available on some url.
    # state_dict = torch.hub.load_state_dict_from_url('url.pth')
    return model

def prepare_model1(ckpt_dir='./ckpt/pretrained.pth', arch='mae_vit_small_patch16'):
    model = getattr(models_mae, arch)(in_chans=1, audio_exp=True, img_size=(1024, 128), decoder_mode=1, decoder_depth=16)
    checkpoint = torch.load(ckpt_dir, map='cpu')
    model.load_state_dict(checkpoint[arch], strict=False)
    return model

def wav2fbank():
    dataset_head = DATASET.take(2)
    dataset_head.cast_column('flac', Audio(sampling_rate=16_000))
    example = next(iter(dataset_head))

    waveform, sr = (example['flac']['array'], example['flac']['sampling_rate'])
    waveform = waveform - waveform.mean()
    waveform = torch.tensor(waveform, device='cpu')
    print(f"waveform shape is {waveform.shape}")
    waveform = waveform.unsqueeze(dim=0)
    print(f"waveform shape is {waveform.shape}")
    fbank = kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                        window_type='hanning', num_mel_bins=MELBINS, dither=0.0, frame_shift=10)
    
    n_frames = fbank.shape[0]
    p = TARGET_LEN - n_frames
    
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, 2))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:TARGET_LEN, :]
    return fbank

def norm_fbank(fbank):
    norm_mean = -4.2677393
    norm_std = 4.5689974
    fbank = (fbank - norm_mean) / (norm_std * 2)
    return fbank

def display_fbank(bank, minmin=None, maxmax=None):
    plt.figure(figsize=(20, 4))
    plt.imshow(20*bank.T.numpy(), origin='lower', interpolation='nearest', vmax=maxmax, vmin=minmin, aspect='auto')

def prepare_fbank():
    fbank = wav2fbank()
    fbank = norm_fbank(fbank)
    display_fbank(fbank)
    
    x = torch.tensor(fbank)
    x = x.unsqueeze(dim=0)
    x = x.unsqueeze(dim=0)
    return x

importlib.reload(models_mae)
model = prepare_model()
x = prepare_fbank()
print("MODEL and DATA LOADED")


model.mask_2d=False
model.mask_t_prob=0.5
model.mask_f_prob=0.3
loss, y, mask, _ = model(x.float(), mask_ratio=0.3)

print(f"y_shape: {y.shape}, mask_shape: {mask.shape}")
print(f'mask sum (masked): {mask.sum()}, mask keep: {np.ones_like(mask).sum()-mask.sum()}')

y = model.unpatchify(y)
y = torch.einsum('nchw->nhwc').detach().cpu()
print('y_unpatchified.shape', y.shape)




# # Basic Inference Setup.
# for name, param in model.named_parameters():
#     print(name, param.shape)
    

## Load audio

## preprocessing the audio using the same pipeline expected.