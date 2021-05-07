import yaml
import argparse

from myaudio import TacotronSTFT
from myprocessor.process import Processor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()
    
    config = yaml.load(open(args.config, 'r'), Loader = yaml.FullLoader)

    processor = Processor(config)
    processor.process_data()

    '''
    !python preprocess.py config/LJSpeech/preprocess.yaml
    '''
