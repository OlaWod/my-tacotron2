import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MyDataset
from model import MyLoss
from utils import to_device


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs):
    preprocess_cfg, model_cfg, train_cfg = configs

    # dataset
    dataset = MyDataset(
        "val.txt", configs
    )

    batch_size = train_cfg["batch_size"]
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )

    # loss
    Loss = MyLoss().to(device)

    # evaluating
    loss_sums = [0] * 5
    
    for batch in loader:
        batch = to_device(batch, device)

        # Forward
        with torch.no_grad():
            output = model(*batch[:-1])

            # Cal loss
            loss = Loss(output, batch)

            for i in range(len(loss)):
                loss_sums[i] += loss[i].item() * len(batch[0])

    # message
    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Gate Loss: {:.4f}, Attention Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )
    
    return message
                    

if __name__ == '__main__':
    import os
    import yaml
    import argparse

    from model import get_model
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=100000)
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

    # Get model
    model = get_model(args, configs, device, train=False)
    
    message = evaluate(model, args.restore_step, configs)
    print(message)
