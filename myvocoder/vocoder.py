import torch
import json
import numpy as np

import hifigan


def get_vocoder(name, device):

    if name == "MelGAN":
        vocoder = torch.hub.load("descriptinc/melgan-neurips", "load_melgan", "multi_speaker")
        vocoder.mel2wav.to(device)
        vocoder.mel2wav.eval()
    elif name == 'MelGAN-LJ':
        vocoder = torch.hub.load("descriptinc/melgan-neurips", "load_melgan", "linda_johnson")
        vocoder.mel2wav.to(device)
        vocoder.mel2wav.eval()
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        ckpt = torch.load("hifigan/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.remove_weight_norm()
        vocoder.to(device)
        vocoder.eval()
    elif name == "HiFi-GAN-LJ":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.remove_weight_norm()
        vocoder.to(device)
        vocoder.eval()
    else: # WaveGlow
        vocoder = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
        vocoder = vocoder.remove_weightnorm(vocoder)
        vocoder = vocoder.to(device)
        vocoder.eval()

    return vocoder


def vocoder_infer(vocoder, mels, vocoder_name, max_wav_value=32768.0):
    with torch.no_grad():
        if vocoder_name == "MelGAN" or vocoder_name == "MelGAN-LJ":
            wavs = vocoder.inverse(mels/np.log(10))
        elif vocoder_name == "HiFi-GAN" or vocoder_name == "HiFi-GAN-LJ":
            wavs = vocoder(mels).squeeze(1)
        else: # WaveGlow
            wavs = vocoder.infer(mels)

    wavs = (max_wav_value * wavs.cpu().numpy()).astype("int16")

    return wavs


if __name__ == '__main__':
    name = "xx"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocoder = get_vocoder(name, device)
    mels = None # (batch, n_mel_channels, n_frames)

    wavs = vocoder_infer(vocoder, mels, name) # (batch, n_sampling_points)
