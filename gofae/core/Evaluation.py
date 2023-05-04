import torch
import sys
import numpy as np
from utilities.TestUtilities import compute_stat_pval
from core.Losses import rec_loss, stat_loss


def validate(nets, dataset, loader, n_z, use_proj, latent_dim_proj, num_projections, use_HT_in_loss, htest,
             lambda_alpha, device):
    with torch.no_grad():
        encoder_p1 = nets[0].eval()
        encoder_p2 = nets[1].eval()
        decoder = nets[2].eval()

        recon_track = []
        stat_track = []
        pval_track = []
        total_loss_track = []

        for batch_idx, all_val_data in enumerate(loader):

            if dataset == 'mnist' or dataset == 'cifar10':
                images = all_val_data[0].to(device)
            elif dataset == 'celeba':
                images = all_val_data.to(device)
            else:
                sys.exit('Dataset not recognized.')

            code_s1 = encoder_p1(images)
            p2_out = encoder_p2(code_s1)
            stat, pval = compute_stat_pval(p2_out, n_z, use_proj, latent_dim_proj, num_projections, htest,
                                           device=device)
            x_recon = decoder(p2_out)
            recon_loss = rec_loss(x_recon, images)
            dist_loss = stat_loss(stat, T_alpha=htest.T_alpha, optdir=htest.optdir, device=device,
                                  use_HT_in_loss=use_HT_in_loss)
            full_loss = recon_loss + lambda_alpha * dist_loss

            recon_track.append(recon_loss.data.item())
            stat_track.append(stat.detach().cpu().numpy())
            pval_track.append(pval)
            total_loss_track.append(full_loss.detach().cpu().numpy())

    return np.mean(np.array(recon_track)), np.mean(np.array(stat_track)), np.mean(np.array(pval_track)), np.mean(
        np.array(total_loss_track))


def evaluate_image_recon(nets, dataset, loader, num_samples, device):

    with torch.no_grad():
        encoder_p1 = nets[0].eval()
        encoder_p2 = nets[1].eval()
        decoder = nets[2].eval()

        store_recons = []
        store_origs = []

        img_cnt = 0
        for batch_idx, all_test_data in enumerate(loader):

            if dataset == 'mnist' or dataset == 'cifar10':
                images = all_test_data[0].to(device)
            elif dataset == 'celeba':
                images = all_test_data.to(device)
            else:
                sys.exit('Dataset not recognized.')

            code_s1 = encoder_p1(images)
            p2_out = encoder_p2(code_s1)
            x_recon = decoder(p2_out)

            store_recons.append(x_recon.view(-1, encoder_p1.n_channel, encoder_p1.img_size, encoder_p1.img_size))
            store_origs.append(images.view(-1, encoder_p1.n_channel, encoder_p1.img_size, encoder_p1.img_size))

            img_cnt += images.shape[0]
            if img_cnt >= num_samples:
                return torch.cat(store_recons, dim=0), torch.cat(store_origs, dim=0)
