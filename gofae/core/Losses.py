import torch
import torch.nn.functional as F


def rec_loss(x_recon, x_true):
    return F.mse_loss(x_recon, x_true, reduction='sum').div(x_recon.shape[0])


def stat_loss(stat, T_alpha, optdir, device, use_HT_in_loss=False):
    if use_HT_in_loss:
        if optdir == 1:
            stat_loss = torch.max(
                torch.tensor(0.0, dtype=torch.float, device=device),
                torch.tensor(T_alpha, dtype=torch.float, device=device) - stat)
        else:
            stat_loss = torch.max(
                torch.tensor(0.0, dtype=torch.float, device=device),
                stat - torch.tensor(T_alpha, dtype=torch.float, device=device))
    else:
        if optdir == 1:
            stat_loss = -stat
        else:
            stat_loss = stat

    return stat_loss