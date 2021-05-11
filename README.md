# my-tacotron2

## Usage

#### LJSpeech

**preprocess:**
```bash
python preprocess.py config/LJSpeech/preprocess.yaml
```

**train:**
```bash
python train.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

**evaluate:**
```bash
python evaluate.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

**synthesize:**
```bash
python synthesize.py --text "has never been surpassed." --restore_step 900 -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

## Alignment

alignment of step 500 - 9000:
![](https://github.com/OlaWod/my-tacotron2/blob/master/alignment_log/alignments.gif)

## Reference

https://arxiv.org/abs/1703.10135

https://arxiv.org/abs/1712.05884?source=post_page-----630afcafb9dd----------------------

https://github.com/NVIDIA/tacotron2

https://github.com/ming024/FastSpeech2