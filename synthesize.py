import os
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from scipy.io import wavfile
import numpy as np
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt

from mytext import process_english
from myvocoder import get_vocoder, vocoder_infer
from model import get_model
from utils import to_device


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    preprocess_cfg, model_cfg, train_cfg = configs

    # model
    model = get_model(args, configs, device, train=False)
    # vocoder
    vocoder = get_vocoder("HiFi-GAN", device)

    # data
    phone_id = torch.LongTensor(process_english(args.text)[-1]).unsqueeze(0).to(device)

    # synthesize
    with torch.no_grad():
        output = model.inference(phone_id)
    synth(args.text, output, vocoder, configs)
        

def synth(text, output, vocoder, configs):
    preprocess_cfg, model_cfg, train_cfg = configs
    sr = preprocess_cfg["audio"]["sampling_rate"]
    result_dir = train_cfg["path"]["result_dir"]
    os.makedirs(result_dir, exist_ok=True)
    
    mel = output[1][0] # (batch, 80, mel_len)
    alignment = output[-1][0]
    
    wav = vocoder_infer(vocoder, mel.unsqueeze(0), "HiFi-GAN")[0]
    wavfile.write(os.path.join(result_dir,"{}.wav".format(text[:10])), sr, wav)

    mel = mel.cpu().numpy().astype(np.float32)
    plt.imshow(mel)
    plt.ylim(0, mel.shape[0])
    plt.colorbar()
    plt.savefig(os.path.join(result_dir, "{}.png".format(text[:10]+"-mel")))
    plt.close()

    alignment = alignment.cpu().numpy().astype(np.float32).T
    plt.imshow(alignment)
    plt.ylim(0, alignment.shape[0])
    plt.colorbar()
    plt.savefig(os.path.join(result_dir, "{}.png".format(text[:10]+"-alignment")))
    plt.close()
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="text to generate",
    )
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

    # synthesize
    main(args, configs)
    
    
