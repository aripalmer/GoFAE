import argparse
import torch
import time
import numpy as np
import pandas as pd
import os
from pathlib import Path
from hypotests.UnifTests import KolmogorovSmirnovUnif
from hypotests.CoreTests import ShapiroWilk, SWF_weights
from core.Manifolds import sample_Stiefel
import yaml


def createETS_Unif(path, htest, test_set_samp_size, num_sims):

    start = time.time()
    empirical_dist_tstat = []

    dist = torch.distributions.uniform.Uniform(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))

    for i in range(num_sims):
        if (i+1) % 5000 == 0:
            print(i+1)
        null_dist_samps = dist.sample(torch.Size([test_set_samp_size]))
        stat = htest.teststat(null_dist_samps)
        empirical_dist_tstat.append(stat.detach().cpu().numpy())
    total = (time.time() - start)/60.
    empirical_dist_tstat = np.sort(np.array(empirical_dist_tstat).reshape(-1,1), axis=0)

    path1 = os.path.join(path, htest.test, str('sample_size_')+str(test_set_samp_size))

    if not os.path.exists(path1):
        print('Saving Statistics for the {} test\n'.format(htest.test))
        Path(path1).mkdir(parents=True, exist_ok=True)

    pd.DataFrame(empirical_dist_tstat).to_csv(os.path.join(path1, str('emp_dist_tstat.csv')) ,header=['tstat'], index=False)

    # Logfile
    txt_file = open(os.path.join(path1, str('logfile_tstat_')+str(htest.test)+str('.txt')), "w")
    txt_file.write("Total number of simulations: %d \n" % num_sims)
    txt_file.write('Time was {:.2f} minutes \n'.format(total))
    txt_file.close()


def createETS1D_Norm(path, htest, n_z, device, num_sims, batch_size, num_projections):

    if num_projections > n_z:
        raise Exception('num_projections must be less than or equal to n_z.')

    print('Using projection 1d to estimate empirical test statistic distribution')
    start = time.time()

    empirical_dist_worst = []
    empirical_dist_avg = []

    # assume initial encoded data has dimension (n x m)
    # univariate will project to (n x 1)
    # multivariate will project to (n x k)
    # samples will come from Stiefel manifold

    # Set parameters for the multivariate normal null distribution
    loc = torch.zeros(n_z, device=device)
    covar = torch.eye(n_z, device=device)
    dist = torch.distributions.multivariate_normal.MultivariateNormal(loc, covar)

    for i in range(num_sims):

        if (i + 1) % 5000 == 0:
            print(i)

        # Get batch_size samples from MVN_m(0, I)
        null_dist_samps = dist.sample(torch.Size([batch_size]))

        # Get m projections to test
        stiefel_sample = sample_Stiefel(n_z, num_projections, device)

        # create temp list
        temp = []

        projected = torch.matmul(null_dist_samps, stiefel_sample)
        for direction in range(num_projections):
            temp_stat = htest.teststat(x=projected[:, direction])
            temp.append(temp_stat.view(-1, 1))
        out = torch.cat(temp, dim=1)

        # statistics to use [min, max]
        if htest.optdir == -1.:
            val = torch.max(out)
        else:
            val = torch.min(out)

        # save out avg
        val_avg = torch.mean(out)

        empirical_dist_worst.append(val.detach().cpu().numpy())
        empirical_dist_avg.append(val_avg.detach().cpu().numpy())

    total = (time.time() - start) / 60.
    empirical_dist_worst = np.sort(np.array(empirical_dist_worst).reshape(-1, 1), axis=0)
    empirical_dist_avg = np.sort(np.array(empirical_dist_avg).reshape(-1, 1), axis=0)

    path1 = os.path.join(path, htest.test, str('latent_dim_')+str(n_z))
    path2 = os.path.join(str('latent_dim_proj_1'), str('batch_size_')+str(batch_size),
                         str('inner_sim_')+str(num_projections))

    if not os.path.exists(os.path.join(path1, path2)):
        print('Saving Statistics for the {} test\n'.format(htest.test))
        Path(os.path.join(path1, path2)).mkdir(parents=True, exist_ok=True)

    if htest.optdir == -1:
        pd.DataFrame(empirical_dist_worst).to_csv(os.path.join(path1, path2, str('emp_dist_max.csv')), header=['max'],
                                                  index=False)
    else:
        pd.DataFrame(empirical_dist_worst).to_csv(os.path.join(path1, path2, str('emp_dist_min.csv')), header=['min'],
                                                  index=False)

    pd.DataFrame(empirical_dist_avg).to_csv(os.path.join(path1, path2, str('emp_dist_avg.csv')), header=['avg'], index=False)

    # Logfile
    txt_file = open(os.path.join(path1, path2, str('logfile.txt')), "w")
    txt_file.write("Total number of simulations: %d \n" % num_sims)
    txt_file.write("Number of projections: %d \n" % num_projections)
    txt_file.write("Projecting from: %d --> %d \n" % (n_z, 1))
    txt_file.write('Time was {:.2f} minutes \n'.format(total))
    txt_file.close()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='Create distribution of a test statistic under the null.')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/example_null.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    if config['make_test']=='gof':
        sw_wts, sw_mu, sw_sigma, _, _, _ = SWF_weights(N=config['batch_size'], device=device)
        hypothesis_test = ShapiroWilk(emp_dist=None, sw_wts=sw_wts, new_mu=sw_mu, new_sigma=sw_sigma, device=device, new_stat=None,
                         T_alpha=None, use_emp=False)
        createETS1D_Norm(path=config['diststat_path'], htest=hypothesis_test, n_z=config['n_z'], device=device, num_sims=config['num_sims'],
                      batch_size=config['batch_size'], num_projections=config['num_projections'])
    elif config['make_test']=='hc':
        hypothesis_test = KolmogorovSmirnovUnif(device=device, emp_dist=None)
        createETS_Unif(path=config['diststat_path'], htest=hypothesis_test, test_set_samp_size=config['hc_test']['test_set_samp_size'], num_sims=config['hc_test']['num_sims'])
    else:
        raise Exception('Not valid')



