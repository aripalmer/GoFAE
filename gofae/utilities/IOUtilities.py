import sys
import numpy as np
import pandas as pd
import torch
import os


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def save_model(dirpath, nets, optimizers, epoch, stats):
    enc1 = nets[0]
    enc2 = nets[1]
    dec = nets[2]
    enc1_optim = optimizers[0]
    enc2_optim = optimizers[1]
    dec_optim = optimizers[2]

    torch.save({
        'epoch': epoch,
        'encoder_p1_state_dict': enc1.state_dict(),
        'encoder_p2_state_dict': enc2.state_dict(),
        'decoder_state_dict': dec.state_dict(),
        'encoder_optimizer_p1_state_dict': enc1_optim.state_dict(),
        'encoder_optimizer_p2_state_dict': enc2_optim.state_dict(),
        'decoder_optimizer_state_dict': dec_optim.state_dict(),
        'running_mean': stats[0],
        'running_cov': stats[1]
    }, dirpath)


def save_data(path, tracked_recon, tracked_tstat, tracked_pval, tracked_total_loss):
    full_data = np.hstack((np.array(tracked_recon).reshape(-1, 1), np.array(tracked_tstat).reshape(-1, 1),
                                 np.array(tracked_pval).reshape(-1, 1),
                                 np.array(tracked_total_loss).reshape(-1, 1)))
    pd.DataFrame(full_data).to_csv(path, header=['recon', 'tstat', 'pval', 'total_loss'], index=False)


def restore_model(path, encoder_p1, encoder_p2, decoder):

    if os.path.isfile(path):
        checkpoint = torch.load(path)
        encoder_p1.load_state_dict(checkpoint['encoder_p1_state_dict'])
        encoder_p2.load_state_dict(checkpoint['encoder_p2_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        running_mean = checkpoint['running_mean']
        running_cov = checkpoint['running_cov']
        return running_mean, running_cov
    else:
        raise Exception("Model does not exist.")


def tuple_constructor(loader, node):
    # Load the sequence of values from the YAML node
    values = loader.construct_sequence(node)
    # Return a tuple constructed from the sequence
    return tuple(values)



