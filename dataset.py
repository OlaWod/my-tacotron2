import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from mytext import phone_to_sequence


class MyDataset(Dataset):
    def __init__(self, metafile, configs):
        preprocess_cfg, model_cfg, train_cfg = configs

        self.feature_dir = preprocess_cfg['path']['feature_dir']
        self.n_frames_per_step = model_cfg['decoder']['n_frames_per_step']
        
        self.basename, self.phone = self.process_meta(metafile)
        
    
    def __getitem__(self, idx):
        basename = self.basename[idx]
        phone = self.phone[idx]
        phone_id = phone_to_sequence(phone, ["english_cleaners"])
        phone_id = torch.IntTensor(phone_id)

        # mel
        mel_path = os.path.join(self.feature_dir, 'mel', 'mel-{}.npy'.format(basename))
        mel = torch.from_numpy(np.load(mel_path))

        return phone_id, mel

  
    def __len__(self):
        return len(self.basename)


    def process_meta(self, metafile):
        with open(os.path.join(self.feature_dir, metafile), 'r', encoding='utf-8') as f:
            basename = []
            phone = []
            
            for line in f.readlines():
                infos = line.strip('\n').split('|')
                basename.append(infos[0])
                phone.append(infos[-1])
                
        return basename, phone


    def collate_fn(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths
    

if __name__ == "__main__":
    import os
    import yaml
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--preprocess_config", type=str, required=True, help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()
    
    # Load Config
    preprocess_cfg = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_cfg = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_cfg = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_cfg, model_cfg, train_cfg)

    dataset = MyDataset("train.txt", configs)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    for batch in loader:
        print(batch)
