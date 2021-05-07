import os
import torch
import numpy as np
import pyworld as pw


def load_wav_to_torch(filename):
    sampling_rate, data = read(filename)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def get_mel_energy_from_wav(audio, sampling_rate, stft, max_wav_value=32768.0):
    audio = torch.FloatTensor(audio.astype(np.float32))
    if sampling_rate != stft.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
        
    audio_norm = audio / max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)

    melspec, energy = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    energy = torch.squeeze(energy, 0)

    return melspec, energy


def get_f0_from_wav(audio, sampling_rate, hop_length=256, max_wav_value=32768.0):
    audio_norm = (audio / max_wav_value).astype(np.float64)

    _f0, t = pw.dio(
            audio_norm,
            sampling_rate,
            frame_period=hop_length / sampling_rate * 1000,
        )
    f0 = pw.stonemask(audio_norm, _f0, t, sampling_rate)
    
    return f0


def get_feature_from_npy(filename):
    feature = torch.from_numpy(np.load(filename))
    return feature


def save_feature_to_npy(feature, feature_type, out_dir='./', basename='xx'):
    npy_name = "{}-{}.npy".format(feature_type, basename)
    np.save(os.path.join(out_dir, npy_name), feature)


if __name__ == '__main__':
    import yaml
    from scipy.io.wavfile import read
    from .stft import TacotronSTFT

    config = yaml.load(open('./config.yaml', 'r'))
    
    _stft = TacotronSTFT(
            config["stft"]["filter_length"],
            config["stft"]["hop_length"],
            config["stft"]["win_length"],
            config["mel"]["n_mel_channels"],
            config["audio"]["sampling_rate"],
            config["mel"]["mel_fmin"],
            config["mel"]["mel_fmax"],
        )

    sampling_rate, audio = read('xx.wav')

    mel, energy = get_mel_energy_from_wav(audio, sampling_rate, _stft)
    f0 = get_f0_from_wav(audio, sampling_rate)
    
    save_feature_to_npy(mel, 'mel')
    save_feature_to_npy(energy, 'energy')
    save_feature_to_npy(f0, 'f0')

    mel_ = get_feature_from_npy('mel-xx.npy')
    energy_ = get_feature_from_npy('energy-xx.npy')
    f0_ = get_feature_from_npy('f0-xx.npy')
