import torch
import torch.nn as nn


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss() # Sigmoid + BCELoss

    def forward(self, pred, truth):
        mel_pred, mel_pred_postnet, stop_pred, _ = pred
        mel_truth, stop_truth = truth[2], truth[3]  

        mel_truth.requires_grad = False
        mel_truth.requires_grad = False

        stop_pred = stop_pred.view(-1, 1)
        stop_truth = stop_truth.view(-1, 1)

        mel_loss = self.mse_loss(mel_pred, mel_truth)
        mel_postnet_loss = self.mse_loss(mel_pred_postnet, mel_truth)
        stop_loss = self.bce_loss(stop_pred, stop_truth)

        total_loss = mel_loss + mel_postnet_loss + stop_loss

        return total_loss, mel_loss, mel_postnet_loss, stop_loss
