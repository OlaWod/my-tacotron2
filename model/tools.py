import os
import torch

from .optimizer import ScheduledOptim
from .model import MyModel


def get_model(args, configs, device, train=False):
    preprocess_cfg, model_cfg, train_cfg = configs

    model = MyModel(configs).to(device)

    restore_step = args.restore_step
    if restore_step:
        ckpt_path = os.path.join(
            train_cfg["path"]["ckpt_dir"],
            "{}.pth.tar".format(restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        optimizer = ScheduledOptim(model, train_cfg, model_cfg)
        if restore_step:
            optimizer.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, optimizer

    model.eval()
    model.requires_grad_ = False
    return model
