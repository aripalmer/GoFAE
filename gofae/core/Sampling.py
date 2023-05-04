import torch
import numpy as np
import os
from pathlib import Path
from torchvision.utils import save_image
from core.Evaluation import evaluate_image_recon


def samples(nets, stats, num_samples=100, device=None):

    with torch.no_grad():
        encoder_p1 = nets[0].eval()
        encoder_p2 = nets[1].eval()
        decoder = nets[2].eval()

        m = torch.tensor(np.random.multivariate_normal(mean=stats[0].reshape(-1), cov=stats[1], size=num_samples),
                             dtype=torch.float, device=device)

        gen_ims = decoder(m).view(num_samples, encoder_p1.n_channel, encoder_p1.img_size, encoder_p1.img_size)
    return gen_ims


def samples_for_fid(save_path, nets, stats, loader, device, normalize, num_samps=10_000):

    with torch.no_grad():
        encoder_p1 = nets[0].eval()
        encoder_p2 = nets[1].eval()
        decoder = nets[2].eval()

        samp_path_is = os.path.join(save_path, 'generated_samples')
        if not os.path.exists(samp_path_is):
            Path(samp_path_is).mkdir(parents=True, exist_ok=True)

        for i in range(num_samps):
            samp_imgs = samples(nets=(encoder_p1, encoder_p2, decoder), stats=(stats[0], stats[1]), num_samples=1, device=device)
            save_image(samp_imgs.view(1, encoder_p1.n_channel, encoder_p1.img_size, encoder_p1.img_size),
                       samp_path_is + str(i) + '.png', normalize=normalize)

        recon_path_is = os.path.join(save_path, 'reconstructions')
        if not os.path.exists(recon_path_is):
            Path(recon_path_is).mkdir(parents=True, exist_ok=True)

        test_recon_imgs, _ = evaluate_image_recon(nets=(encoder_p1, encoder_p2, decoder), loader=loader, num_samples=num_samps)
        for i in range(num_samps):
            save_image(test_recon_imgs[i].view(1, encoder_p1.n_channel, encoder_p1.img_size, encoder_p1.img_size),
                       recon_path_is + str(i) + '.png', normalize=normalize)






