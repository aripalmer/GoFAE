import sys
import numpy as np
import torch
from hypotests.CoreTests import AndersonDarling, CramerVonMises, KolmogorovSmirnov, \
    ShapiroFrancia, ShapiroWilk, HenzeZirkler, MardiaSkew, Royston, EppsPulley1
from core.Manifolds import sample_Stiefel
from hypotests.UnifTests import KolmogorovSmirnovUnif


def testsetup(test, emppath, test_dictionary):
    if test == 'hz':
        return HenzeZirkler(emppath, test_dictionary["device"], test_dictionary["new_stat"], test_dictionary["n"],
                            test_dictionary["n_z"], test_dictionary["T_alpha"], test_dictionary["use_emp"])
    elif test == 'sw':
        return ShapiroWilk(emppath, test_dictionary["sw_wts"], test_dictionary["sw_mu"],
                           test_dictionary["sw_sigma"], test_dictionary["device"], test_dictionary["new_stat"], test_dictionary["T_alpha"], test_dictionary["use_emp"])
    elif test == 'sf':
        return ShapiroFrancia(emppath, test_dictionary["sf_wts"], test_dictionary["sf_mu"], test_dictionary["sf_sigma"],
                              test_dictionary["device"], test_dictionary["new_stat"], test_dictionary["T_alpha"], test_dictionary["use_emp"])
    elif test == 'ad':
        return AndersonDarling(emppath, test_dictionary["n"], test_dictionary["device"], test_dictionary["new_stat"], test_dictionary["T_alpha"], test_dictionary["use_emp"])
    elif test == 'cvm':
        return CramerVonMises(emppath, test_dictionary["n"], test_dictionary["device"], test_dictionary["new_stat"], test_dictionary["T_alpha"], test_dictionary["use_emp"])
    elif test == 'ep1':
        return EppsPulley1(emppath, test_dictionary["n"], test_dictionary["device"], test_dictionary["new_stat"], test_dictionary["T_alpha"], test_dictionary["use_emp"])
    elif test == 'ks':
        return KolmogorovSmirnov(emppath, test_dictionary["n"], test_dictionary["device"], test_dictionary["new_stat"], test_dictionary["T_alpha"], test_dictionary["use_emp"])
    elif test == 'mardia_skew':
        return MardiaSkew(emppath, test_dictionary["n_z"], test_dictionary["device"], test_dictionary["new_stat"], test_dictionary["T_alpha"], test_dictionary["use_emp"])
    elif test == 'royston':
        sw=ShapiroWilk(emppath, test_dictionary["sw_wts"], test_dictionary["sw_mu"],
                    test_dictionary["sw_sigma"], test_dictionary["device"], test_dictionary["new_stat"], test_dictionary["T_alpha"], test_dictionary["use_emp"])
        sf=ShapiroFrancia(emppath, test_dictionary["sf_mu"], test_dictionary["sf_sigma"],
                       test_dictionary["device"], test_dictionary["new_stat"], test_dictionary["T_alpha"], test_dictionary["use_emp"])
        return Royston(emppath, test_dictionary["e"], test_dictionary["device"], sf,
                       sw, test_dictionary["new_stat"], test_dictionary["n"], test_dictionary["T_alpha"], test_dictionary["use_emp"])
    else:
        sys.exit("Test not implemented")


def check_test(test, latent_dim_proj):
    if not test.is_univariate():
        if latent_dim_proj == 1:
            raise Exception("Project to higher dimension or select a univariate test.")
        else:
            pass
    else:
        if latent_dim_proj > 1:
            raise Exception("Project to 1-dimeision or select a multivariate test.")
        else:
            pass


def compute_mi_cn(n_z, nets, stats, device, num_samps=10_000, chunk_size=200):
    with torch.no_grad():
        encoder_p1 = nets[0].eval()
        encoder_p2 = nets[1].eval()
        decoder = nets[2].eval()

        y_mean = stats[0]
        y_cov = stats[1]

        store_y = np.empty((0, n_z), float)
        store_y_code = np.empty((0, n_z), float)

        for i in range(num_samps//chunk_size):

            y = np.random.multivariate_normal(mean=y_mean.reshape(-1), cov=y_cov, size=chunk_size)
            m = torch.tensor(y, dtype=torch.float, device=device)
            gen_imgs = decoder(m).view(chunk_size, encoder_p1.n_channel, encoder_p1.img_size, encoder_p1.img_size)
            code_s1 = encoder_p1(gen_imgs)
            y_code = encoder_p2(code_s1)

            y_code = y_code.detach().cpu().numpy()
            store_y = np.append(store_y, y, axis=0)
            store_y_code = np.append(store_y_code, y_code, axis=0)

        joint = np.hstack((store_y, store_y_code))
        sig1 = np.cov(store_y, rowvar=False)
        sig2 = np.cov(store_y_code, rowvar=False)
        sig_joint = np.cov(joint, rowvar=False)
        mutual_info = 0.5*np.log((np.linalg.det(sig1)*np.linalg.det(sig2))/np.linalg.det(sig_joint))
        cond_num = np.linalg.cond(y_cov)

        return mutual_info, cond_num


def compute_stat_pval(p2_out, n_z, use_proj, latent_dim_proj, num_projections, htest, device):

    if htest.is_univariate():
        temp = []
        stiefel_sample = sample_Stiefel(n_z, num_projections, device)
        projected = torch.matmul(p2_out, stiefel_sample)
        for direction in range(num_projections):
            temp_stat = htest.teststat(x=projected[:, direction])
            temp.append(temp_stat.view(-1, 1))
        # Option for multivariate tests
    else:  # is multivariate
        # multivariate tests can be projected
        if use_proj:
            temp = []
            for direction in range(num_projections):
                stiefel_sample = sample_Stiefel(n_z, latent_dim_proj, device)
                projected = torch.matmul(p2_out, stiefel_sample)
                if isinstance(htest, Royston):
                    temp_stat, H_e = htest.teststat(x=projected)
                else:
                    temp_stat = htest.teststat(x=projected)
                temp.append(temp_stat.view(-1, 1))
        # multivariate tests don't need to be projected
        else:
            if isinstance(htest, Royston):
                temp_stat, H_e = htest.teststat(x=p2_out)
            else:
                temp_stat = htest.teststat(x=p2_out)

        #############################################
        # Calculate final test statistic #
        #############################################
    if htest.is_maximization() and htest.new_stat == 'min':
        out_tensor = torch.cat(temp, dim=1)
        stat = torch.min(out_tensor)
    elif htest.is_maximization() and htest.new_stat == 'avg':
        out_tensor = torch.cat(temp, dim=1)
        stat = torch.mean(out_tensor)
    elif not htest.is_maximization() and htest.new_stat == 'max':
        out_tensor = torch.cat(temp, dim=1)
        stat = torch.max(out_tensor)
    elif not htest.is_maximization() and htest.new_stat == 'avg':
        out_tensor = torch.cat(temp, dim=1)
        stat = torch.mean(out_tensor)
    elif not htest.is_univariate and htest.new_stat == 'none':
        stat = temp_stat
    else:
        sys.exit("Configuration incorrect for test statistic")

    pval = htest.pval(stat)
    return stat, pval


def evaluate_normality(dataset, nets, loader, n_z, use_proj, latent_dim_proj, num_projections, htest, test_set_samp_size, device):
    with torch.no_grad():
        encoder_p1 = nets[0].eval()
        encoder_p2 = nets[1].eval()
        stat_track = []
        pval_track = []

        batch_cnt = 0

        for epoch in range((test_set_samp_size // len(loader))+1):
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
                stat_track.append(stat.detach().cpu().numpy())
                pval_track.append(pval)
                batch_cnt += 1

                if batch_cnt >= test_set_samp_size:
                    return np.array(stat_track).reshape(-1, 1), np.array(pval_track).reshape(-1, 1)


def evaluate_uniformity(trained_gof, dataset, diststat_path, test_set_samp_size, nets, loader, n_z, use_proj,
                        latent_dim_proj, num_projections, htest_map, device, num_repeats=30, eval_all=False):

    with torch.no_grad():
        encoder_p1 = nets[0].eval()
        encoder_p2 = nets[1].eval()

        ks_unif_emp_dist = KolmogorovSmirnovUnif.get_ks_unif_info(diststat_path, test_set_samp_size)
        ks_htest = KolmogorovSmirnovUnif(device, ks_unif_emp_dist)

        store_KS_info = []
        for test in htest_map:

            if (not eval_all and test == trained_gof) or eval_all:

                for sim in range(num_repeats):

                    eval_stat, eval_pval = evaluate_normality(dataset=dataset, nets=(encoder_p1, encoder_p2), loader=loader,
                                                                   n_z=n_z, use_proj=use_proj,
                                                                   latent_dim_proj=latent_dim_proj,
                                                                   num_projections=num_projections,
                                                                   htest=htest_map[test],
                                                                   test_set_samp_size=test_set_samp_size,
                                                                   device=device)

                    ks_unif_tstat = ks_htest.teststat(
                        torch.tensor(eval_pval, dtype=torch.float, device=device))
                    ks_unif_pval = ks_htest.pval(ks_unif_tstat)
                    store_KS_info.append([ks_unif_tstat.detach().cpu().numpy(), ks_unif_pval, trained_gof, test])

    return store_KS_info


