import numpy as np
import os
import pandas as pd
import sys
import time
import torch.optim as optim
from pathlib import Path
from torchvision.utils import save_image
from hypotests.CoreTests import SWF_weights
from utilities.TestUtilities import check_test, testsetup, compute_mi_cn, compute_stat_pval, evaluate_uniformity
from core.Manifolds import replacegrad, retract2manifold, sample_Stiefel
from core.Sampling import samples, samples_for_fid
from utilities.IOUtilities import save_model, eprint
from architecture.Encoders import EncoderP2, EncoderMNISTP1, EncoderCelebaP1, EncoderCifar10P1
from architecture.Decoders import DecoderMNIST, DecoderCeleba, DecoderCifar10
from core.Dataset import create_dataloaders, create_test_set_shuffle
from utilities.IOUtilities import save_data
from core.Losses import rec_loss, stat_loss
from core.Evaluation import evaluate_image_recon, validate
from utilities.IOUtilities import restore_model
import yaml


class GoFAE():

    def __init__(self, device, args):

        self.device = device
        self.uvn = {'sw', 'sf', 'ad', 'cvm', 'ks', 'ep1', 'ep2'}
        self.mvn = {'hz', 'mardia_skew', 'royston'}
        self.min_test_stats = {'hz', 'ad', 'cvm', 'mardia_skew', 'royston', 'ks', 'ep2'}
        self.max_test_stats = {'sf', 'sw', 'ep1'}
        self.test = args['gof_test_params']['test']
        self.new_stat = args['gof_test_params']['new_stat']
        self.use_proj = args['gof_test_params']['use_proj']
        self.n_z = args['gof_test_params']['n_z']
        self.latent_dim_proj = args['gof_test_params']['latent_dim_proj']
        self.num_projections = args['gof_test_params']['num_projections']
        self.batch_size = args['gof_test_params']['batch_size']
        self.alpha = args['gof_test_params']['alpha']
        self.epochs = args['trainer_params']['epochs']
        self.lambda_alpha = args['gof_test_params']['lambda_alpha']
        self.num_workers = args['trainer_params']['num_workers']
        self.test_set_samp_size = args['hc_test_params']['test_set_samp_size']
        self.use_HT_in_loss = args['gof_test_params']['use_HT_in_loss']
        self.normalize_img = args['logging_params']['normalize_img']
        self.dataset = args['data_params']['dataset']
        self.lr_adam_enc_p1 = args['optim_params']['lr_adam_enc_p1']
        self.beta1_enc_p1 = args['optim_params']['beta1_enc_p1']
        self.beta2_enc_p1 = args['optim_params']['beta2_enc_p1']
        self.lr_cycle = args['optim_params']['lr_cycle']
        self.lr_sgd = args['optim_params']['lr_sgd']
        self.lr_adam_dec = args['optim_params']['lr_adam_dec']
        self.beta1_dec = args['optim_params']['beta1_dec']
        self.beta2_dec = args['optim_params']['beta2_dec']
        self.momentum = args['gof_test_params']['momentum']
        self.ncv = args['logging_params']['ncv']
        self.experiment = args['logging_params']['experiment']
        self.output_path = args['logging_params']['output_path']
        self.diststat_path = args['gof_test_params']['diststat_path']
        self.data_path = args['data_params']['data_path']
        self.gen_samples = args['data_params']['gen_samples']
        self.num_repeats = args['hc_test_params']['num_repeats']
        self.train = args['trainer_params']['train']
        self.eval_tests = args['eval_tests']

        self.path = os.path.join(self.output_path, str('experiment_')+str(self.experiment))

        if not os.path.exists(self.path):
            Path(self.path).mkdir(parents=True, exist_ok=True)

        if self.train:
            with open(os.path.join(self.path, 'training_config.yaml'), 'w') as file:
                yaml.dump(args, file)

        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(dataset=self.dataset, root_folder=self.data_path,
                                                                                  batch_size=self.batch_size, num_workers=self.num_workers)
        self.test_loader_shuffle = create_test_set_shuffle(dataset=self.dataset, root_folder=self.data_path,
                                                           batch_size=self.batch_size, num_workers=self.num_workers)

        self.test_tuple = [(self.test, self.use_proj, self.new_stat)]

    def get_stat_info(self, path, use_proj, test, n_z, batch_size, latent_dim_proj, num_projections, new_stat):
        # path: specifies path to empirical distribution information
        # use_proj: whether or not projection is used
        # test: which test to use (see possiblities)
        # n_z: dimension of latent variable
        # batch_size: size of batch, needed for calculating correct test statistic and p-values
        # latent_dim_proj: the dimensionality of the latent space after projecting it (for univariate tests, project to 1d)
        # num_projections: how many projections will be used when calculating the new statistic
        # alpha: if no projections are used, then alpha is needed to select the correct T_alpha (critical value)

        if use_proj:

            emp_dist_path = os.path.join(self.diststat_path, str(test), 'latent_dim_'+str(n_z),
                                         'latent_dim_proj_'+str(latent_dim_proj),
                                         'batch_size_'+str(self.batch_size), 'inner_sim_'+str(num_projections))

            if test in self.max_test_stats and new_stat == 'min' and os.path.isfile(os.path.join(emp_dist_path, 'emp_dist_min.csv')):
                #if Path(path).is_file(os.path.join(emp_dist_path, 'emp_dist_min.csv')):
                emp_dist_path1 = os.path.join(emp_dist_path, 'emp_dist_min.csv')  # select path for the min empirical dist
                emp_dist = np.array(pd.read_csv(emp_dist_path1))  # import min emp dist
                T_alpha = np.quantile(emp_dist, self.alpha).astype(np.float32)  # Grab the correct T_alpha
                return emp_dist, T_alpha
            elif test in self.max_test_stats and new_stat == 'avg' and os.path.isfile(os.path.join(emp_dist_path, 'emp_dist_avg.csv')):
                emp_dist_path1 = os.path.join(emp_dist_path, 'emp_dist_avg.csv')  # select path for the avg empirical dist
                emp_dist = np.array(pd.read_csv(emp_dist_path1))  # import avg emp dist
                T_alpha = np.quantile(emp_dist, self.alpha).astype(np.float32)  # Grab the correct T_alpha
                return emp_dist, T_alpha
            elif test in self.min_test_stats and new_stat == 'max' and os.path.isfile(os.path.join(emp_dist_path, 'emp_dist_max.csv')):
                emp_dist_path1 = os.path.join(emp_dist_path, 'emp_dist_max.csv')  # select path for the max empirical dist
                emp_dist = np.array(pd.read_csv(emp_dist_path1))  # import max emp dist
                T_alpha = np.quantile(emp_dist, 1 - self.alpha).astype(np.float32)  # Grab the correct T_alpha
                return emp_dist, T_alpha
            elif test in self.min_test_stats and new_stat == 'avg' and os.path.isfile(os.path.join(emp_dist_path, 'emp_dist_avg.csv')):
                emp_dist_path1 = os.path.join(emp_dist_path, 'emp_dist_avg.csv')  # select path for the avg empirical dist
                emp_dist = np.array(pd.read_csv(emp_dist_path1))  # import avg emp dist
                T_alpha = np.quantile(emp_dist, 1. - self.alpha).astype(np.float32)  # Grab the correct T_alpha
                return emp_dist, T_alpha
            else:
                raise NotImplementedError("Test has not been created, check configuration or create it.")
        else:
            if test in self.mvn and new_stat == 'none' and os.path.isfile(os.path.join(self.diststat_path, str(test), 'latent_dim_' + str(n_z),
                                           'emp_cv_alpha_'+str(self.alpha) + '.csv')):
                emp_cv_path = os.path.join(self.diststat_path, str(test), 'latent_dim_' + str(n_z),
                                           'emp_cv_alpha_'+str(self.alpha) + '.csv')
                T_alpha = np.array(pd.read_csv(emp_cv_path))[0]
                return None, T_alpha
            else:
                raise NotImplementedError("Test has not been created, check configuration or create it.")

    def run(self):
        encoder_p1, encoder_p2, decoder = self.create_models()
        enc_p1_optim, enc_p2_optim, dec_optim = self.create_optimizers(encoder_p1, encoder_p2, decoder)
        enc_p2_scheduler = optim.lr_scheduler.OneCycleLR(enc_p2_optim, self.lr_cycle, epochs=self.epochs,
                                                         steps_per_epoch=len(self.train_loader))
        if self.dataset == 'cifar10':
            enc_p1_scheduler = optim.lr_scheduler.ReduceLROnPlateau(enc_p1_optim, mode='min', factor=0.2, patience=10,
                                                                    min_lr=5e-5, verbose=True)
            dec_scheduler = optim.lr_scheduler.ReduceLROnPlateau(dec_optim, mode='min', factor=0.2, patience=10,
                                                                 min_lr=5e-5, verbose=True)

        htest_map = self.generate_test_map(tests=self.test_tuple)

        if self.use_proj:
            check_test(htest_map[self.test], self.latent_dim_proj)

        track_train_recon = []
        track_train_tstat = []
        track_train_pval = []
        track_train_total_loss = []
        track_val_recon = []
        track_val_tstat = []
        track_val_pval = []
        track_val_total_loss = []

        start_time = time.time()

        running_mean = np.zeros((self.n_z, 1))
        running_cov = np.zeros((self.n_z, self.n_z))

        for epoch in range(self.epochs):
            step = 0
            temp_train_recon = []
            temp_train_tstat = []
            temp_train_pval = []
            temp_train_total_loss = []

            for batch_idx, all_data in enumerate(self.train_loader):

                if self.dataset == 'mnist' or self.dataset == 'cifar10':
                    images = all_data[0].to(self.device)
                elif self.dataset == 'celeba':
                    images = all_data.to(self.device)
                else:
                    sys.exit('Dataset not recognized.')

                encoder_p1.zero_grad()
                encoder_p2.zero_grad()
                decoder.zero_grad()

                code_s1 = encoder_p1(images)
                p2_out = encoder_p2(code_s1)

                code_mean = np.mean(p2_out.detach().cpu().numpy(), axis=0).reshape(-1, 1)
                code_cov = np.cov(p2_out.detach().cpu().numpy(), rowvar=False).reshape(self.n_z, self.n_z)
                running_mean = (1. - self.momentum) * running_mean + self.momentum * code_mean
                running_cov = (1. - self.momentum) * running_cov + self.momentum * code_cov

                stat, pval = compute_stat_pval(p2_out, n_z=self.n_z, use_proj=self.use_proj, latent_dim_proj=self.use_proj,
                                                    num_projections=self.num_projections, htest=htest_map[self.test], device=self.device)

                x_recon = decoder(p2_out)
                recon_loss = rec_loss(x_recon, images)
                dist_loss = stat_loss(stat, T_alpha=htest_map[self.test].T_alpha, optdir=htest_map[self.test].optdir, device=self.device,
                                      use_HT_in_loss=self.use_HT_in_loss)

                full_loss = recon_loss + self.lambda_alpha*dist_loss
                full_loss.backward()

                temp_train_recon.append(recon_loss.data.item())
                temp_train_tstat.append(stat.data.item())
                temp_train_pval.append(pval)
                temp_train_total_loss.append(full_loss.data.item())

                # Compute Riemannian gradients
                replacegrad(encoder_p2)
                enc_p1_optim.step()
                enc_p2_optim.step()
                dec_optim.step()

                # Retract back to Stiefel manifold
                retract2manifold(encoder_p2)
                enc_p2_scheduler.step()

                if (step + 1) % 100 == 0:
                    print(
                        "Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f, %s: %.4f, P-value: %.4f" %
                        (epoch + 1, self.epochs, step + 1, len(self.train_loader), recon_loss.data.item(), self.test.upper(),
                         stat.data.item(), pval), flush=True)
                step += 1

            track_train_recon.append(np.mean(np.array(temp_train_recon)))
            track_train_tstat.append(np.mean(np.array(temp_train_tstat)))
            track_train_pval.append(np.mean(np.array(temp_train_pval)))
            track_train_total_loss.append(np.mean(np.array(temp_train_total_loss)))

            if (epoch + 1) % self.ncv == 0:

                val_rec, val_tstat, val_pval, val_total_loss = validate(nets=(encoder_p1, encoder_p2, decoder),
                                                                             dataset=self.dataset,
                                                                             loader=self.val_loader,
                                                                             n_z=self.n_z,
                                                                             use_proj=self.use_proj,
                                                                             latent_dim_proj=self.latent_dim_proj,
                                                                             num_projections=self.num_projections,
                                                                             use_HT_in_loss=self.use_HT_in_loss,
                                                                             htest=htest_map[self.test],
                                                                             lambda_alpha=self.lambda_alpha,
                                                                             device=self.device)

                track_val_recon.append(val_rec)
                track_val_tstat.append(val_tstat)
                track_val_pval.append(val_pval)
                track_val_total_loss.append(val_total_loss)
                print('Validation Set:  Recon: {:.4f}, Tstat: {:.4f}, P-value: {:.4f}\n'.format(val_rec, val_tstat,
                                                                                                val_pval), flush=True)

            encoder_p1.train()
            encoder_p2.train()
            decoder.train()

            # Use validation loss to modify learning rate on TRAINING SET
            if self.dataset == 'cifar10':
                enc_p1_scheduler.step(val_total_loss)
                dec_scheduler.step(val_total_loss)

        finish_time = time.time()
        total_time = finish_time - start_time
        save_data(os.path.join(self.path, str('training.csv')), track_train_recon, track_train_tstat, track_train_pval, track_train_total_loss)
        save_data(os.path.join(self.path, str('validation.csv')), track_val_recon, track_val_tstat, track_val_pval, track_val_total_loss)

        save_model(dirpath=os.path.join(self.path, 'model.pt'), nets=(encoder_p1, encoder_p2, decoder),
                      optimizers=(enc_p1_optim, enc_p2_optim, dec_optim), epoch=self.epochs + 1,
                      stats=(running_mean, running_cov))

        eprint('Total time for {} epochs is {:.2f} minutes'.format(self.epochs, total_time / 60.), flush=True)

        txt_file = open(os.path.join(self.path, str('time.txt')), "w")
        txt_file.write("Time was %.2f minutes for %d epochs\n" % (total_time / 60., self.epochs))
        txt_file.close()

        store_KS_info = evaluate_uniformity(trained_gof=self.test, dataset=self.dataset, diststat_path=self.diststat_path, test_set_samp_size=self.test_set_samp_size,
                                            nets=(encoder_p1, encoder_p2), loader=self.train_loader, n_z=self.n_z, use_proj=self.use_proj,
                                            latent_dim_proj=self.latent_dim_proj, num_projections=self.num_projections, htest_map=htest_map,
                                            device=self.device, num_repeats=self.num_repeats, eval_all=False)

        pd.DataFrame(store_KS_info).to_csv(os.path.join(self.path, str('store_KS_info_run.csv')),
                                          header=['KS_tstat', 'KS_pval', 'Trained_with', 'Evaluated_with'], index=False)

        mutual_info, cond_num = compute_mi_cn(n_z=self.n_z, nets=(encoder_p1, encoder_p2, decoder), stats=(running_mean, running_cov),
                                              device=self.device, num_samps=10_000, chunk_size=200)
        pd.DataFrame(np.array([mutual_info, cond_num]).reshape(-1, 2)).to_csv(os.path.join(self.path, str('mi_cn.csv')), header=['mi', 'cn'], index=False)

    def generate_test_map(self, tests):
        test_map = {}

        if not tests:
            raise Exception("No tests have been defined.")

        for test in tests:

            test_dictionary = self.get_test_param_dict(self.device, test[0], test[2])
            emp_dist, T_alpha = self.get_stat_info(path=self.diststat_path, use_proj=test[1], test=test[0],
                                                   n_z=self.n_z, batch_size=self.batch_size,
                                                   latent_dim_proj=self.latent_dim_proj,
                                                   num_projections=self.num_projections, new_stat=test[2])
            test_dictionary.update({"T_alpha": T_alpha})
            hypothesis_test = testsetup(test[0], emp_dist, test_dictionary)
            test_map[test[0]] = hypothesis_test

        return test_map

    def get_test_param_dict(self, device, test, new_stat):
        test_dictionary={}
        if test == 'sw':
            sw_wts, sw_mu, sw_sigma, _, _, _ = SWF_weights(self.batch_size, self.device)
            test_dictionary["sw_wts"]=sw_wts
            test_dictionary["sw_mu"]=sw_mu
            test_dictionary["sw_sigma"]=sw_sigma
        elif test == 'sf':
            _, _, _, sf_wts, sf_mu, sf_sigma = SWF_weights(self.batch_size, self.device)
            test_dictionary["sf_wts"]=sf_wts
            test_dictionary["sf_mu"]=sf_mu
            test_dictionary["sf_sigma"]=sf_sigma
        elif test == 'royston':
            sw_wts, sw_mu, sw_sigma, sf_wts, sf_mu, sf_sigma = SWF_weights(self.batch_size, self.device)
            test_dictionary["sw_wts"] = sw_wts
            test_dictionary["sw_mu"] = sw_mu
            test_dictionary["sw_sigma"] = sw_sigma
            test_dictionary["sf_wts"] = sf_wts
            test_dictionary["sf_mu"] = sf_mu
            test_dictionary["sf_sigma"] = sf_sigma
        else:
            pass
        test_dictionary["device"] = device
        test_dictionary["n_z"] = self.n_z
        test_dictionary["n"] = self.batch_size
        test_dictionary["use_emp"] = self.use_proj
        test_dictionary["new_stat"] = new_stat
        return test_dictionary

    def create_models(self):
        if self.dataset == 'mnist':
            encoder_p1 = EncoderMNISTP1(n_channel=1, dim_h=128, dim_v=256, dim_y=self.n_z).to(self.device)
            encoder_p2 = EncoderP2(dim_v=256, dim_y=self.n_z).to(self.device)
            decoder = DecoderMNIST(n_channel=1, dim_h=128, dim_y=self.n_z).to(self.device)
        elif self.dataset == 'celeba':
            encoder_p1 = EncoderCelebaP1(n_channel=3, dim_h=128, dim_v=256, dim_y=self.n_z).to(self.device)
            encoder_p2 = EncoderP2(dim_v=256, dim_y=self.n_z).to(self.device)
            decoder = DecoderCeleba(n_channel=3, dim_h=128, dim_y=self.n_z).to(self.device)
        elif self.dataset == 'cifar10':
            encoder_p1 = EncoderCifar10P1(n_channel=3, dim_h=64, dim_v=256, dim_y=self.n_z).to(self.device)
            encoder_p2 = EncoderP2(dim_v=256, dim_y=self.n_z).to(self.device)
            decoder = DecoderCifar10(n_channel=3, dim_h=64, dim_y=self.n_z).to(self.device)
        else:
            raise NotImplementedError("Architecture for dataset does not exist.")
        return encoder_p1, encoder_p2, decoder

    def create_optimizers(self, encoder_p1, encoder_p2, decoder):
        enc_p1_optim = optim.Adam(encoder_p1.parameters(), lr=self.lr_adam_enc_p1,
                                  betas=(self.beta1_enc_p1, self.beta2_enc_p1))
        enc_p2_optim = optim.SGD(encoder_p2.parameters(), lr=self.lr_sgd)
        dec_optim = optim.Adam(decoder.parameters(), lr=self.lr_adam_dec, betas=(self.beta1_dec, self.beta2_dec))
        return enc_p1_optim, enc_p2_optim, dec_optim

    def restore_eval(self):

        htest_map = self.generate_test_map(tests=self.eval_tests)
        encoder_p1, encoder_p2, decoder = self.create_models()
        encoder_p1, encoder_p2, decoder = encoder_p1.to(self.device), encoder_p2.to(self.device), decoder.to(self.device)
        running_mean, running_cov = restore_model(os.path.join(self.path, 'model.pt'), encoder_p1, encoder_p2, decoder)

        test_rec_mse, *_ = validate(nets=(encoder_p1, encoder_p2, decoder), dataset=self.dataset, loader=self.test_loader, n_z=self.n_z,
                                         use_proj=self.use_proj, latent_dim_proj=self.latent_dim_proj,
                                         num_projections=self.num_projections, use_HT_in_loss=self.use_HT_in_loss, htest=htest_map[self.test],
                                         lambda_alpha=self.lambda_alpha, device=self.device)

        save_img_path = os.path.join(self.path, 'img_output')
        if not os.path.exists(save_img_path):
             Path(save_img_path).mkdir(parents=True, exist_ok=True)

        recon_test_imgs, orig_test_imgs = evaluate_image_recon(nets=(encoder_p1, encoder_p2, decoder),
                                                               dataset=self.dataset,
                                                               loader=self.test_loader, num_samples=100,
                                                               device=self.device)

        save_image(
            recon_test_imgs[:100, :, :, :].view(100, encoder_p1.n_channel, encoder_p1.img_size, encoder_p1.img_size),
            os.path.join(save_img_path, str('reconstructed_epoch_') + str(self.epochs + 1) + str('.png')), nrow=10,
            normalize=self.normalize_img)

        save_image(
            orig_test_imgs[:100, :, :, :].view(100, encoder_p1.n_channel, encoder_p1.img_size, encoder_p1.img_size),
            os.path.join(save_img_path, str('ground_truth.png')), nrow=10, normalize=self.normalize_img)

        samp_imgs = samples(nets=(encoder_p1, encoder_p2, decoder), stats=(running_mean, running_cov), num_samples=100,
                            device=self.device)

        save_image(samp_imgs[:100, :, :, :].view(100, encoder_p1.n_channel, encoder_p1.img_size, encoder_p1.img_size),
                   os.path.join(save_img_path, str('generated_epoch_') + str(self.epochs + 1) + str('.png')), nrow=10,
                   normalize=self.normalize_img)

        txt_file = open(os.path.join(self.path, str('logfile.txt')), "w")
        txt_file.write("Test Set MSE: %.4f\n" % test_rec_mse)
        txt_file.close()

        store_KS_info = evaluate_uniformity(trained_gof=self.test, dataset=self.dataset, diststat_path=self.diststat_path, test_set_samp_size=self.test_set_samp_size,
                                            nets=(encoder_p1, encoder_p2), loader=self.test_loader_shuffle, n_z=self.n_z, use_proj=self.use_proj,
                                            latent_dim_proj=self.latent_dim_proj, num_projections=self.num_projections, htest_map=htest_map,
                                            device=self.device, num_repeats=self.num_repeats, eval_all=True)

        pd.DataFrame(store_KS_info).to_csv(os.path.join(self.path, str('store_KS_info_test.csv')),
                                          header=['KS_tstat', 'KS_pval', 'Trained_with', 'Evaluated_with'], index=False)

        if self.gen_samples:
            samples_for_fid(save_path=self.path, nets=(encoder_p1, encoder_p2, decoder), stats=(running_mean, running_cov),
                            loader=self.test_loader, device=self.device, normalize=self.normalize_img, num_samps=10_000)
