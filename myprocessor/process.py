import os
import tgt
import json
import random
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import read

import myaudio
from mytext import _clean_text, process_english


class Processor:
    def __init__(self, config):
        self.config = config

        self.feature_dir = config['path']['feature_dir']
        self.corpus_dir = config['path']['corpus_dir']
        self.wav_dir = os.path.join(config['path']['corpus_dir'], 'wavs')
        self.mel_dir = os.path.join(config['path']['feature_dir'], 'mel')
        
        self.max_wav_value = config['audio']['max_wav_value']
        self.hop_length = config['stft']['hop_length']
        self.sampling_rate = config["audio"]["sampling_rate"]
        self.val_size = config['val_size']

        self.cleaners = config["text"]["text_cleaner"]

        print("Loading STFT...")
        self.stft = myaudio.TacotronSTFT(
            config["stft"]["filter_length"],
            config["stft"]["hop_length"],
            config["stft"]["win_length"],
            config["mel"]["n_mel_channels"],
            config["audio"]["sampling_rate"],
            config["mel"]["mel_fmin"],
            config["mel"]["mel_fmax"],
        )
    
    
    def process_data(self):
        os.makedirs(self.mel_dir, exist_ok=True)

        infos = []
        speakers = {}
        print('Start processing...')

        with open(os.path.join(self.corpus_dir, "metadata.csv"), encoding="utf-8") as f:
            for line in tqdm(f.readlines()):
                parts = line.strip().split("|")
                basename = parts[0]
                text = parts[2]
                text = _clean_text(text, self.cleaners)
                phone = process_english(text)[0]

                wav_path = os.path.join(self.wav_dir, basename+'.wav')
                if not os.path.exists(wav_path):
                    continue

                # get features
                mel = self.get_mel(wav_path)

                info = "|".join([basename, text, phone])
                infos.append(info)
                
                # save
                self.save_npys(mel, basename)
        
        self.save_infos(infos)
        

    def save_npys(self, mel, basename):
        myaudio.save_feature_to_npy(mel, 'mel', self.mel_dir, basename)


    def save_infos(self, infos):
        random.shuffle(infos)

        with open(os.path.join(self.feature_dir, "train.txt"), "w", encoding="utf-8") as f:
            for info in infos[self.val_size :]:
                f.write(info + "\n")
        with open(os.path.join(self.feature_dir, "val.txt"), "w", encoding="utf-8") as f:
            for info in infos[: self.val_size]:
                f.write(info + "\n")
            

    def get_mel(self, wav_path):
        # wav
        sampling_rate, wav = read(wav_path)

        # mel
        mel, energy = myaudio.get_mel_energy_from_wav(wav, sampling_rate, self.stft, self.max_wav_value)
        mel = mel.numpy().astype(np.float32)

        # return
        return mel


