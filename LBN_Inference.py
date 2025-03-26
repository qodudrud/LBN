import argparse
import json
import os
import copy

import numpy as np
import scipy
from scipy import stats
import pandas as pd
import torch

from model.net import LNN_Drift, LNN_Diff, LBNN_Drift, LBNN_Diff
from model.train import train, evaluate

from misc.loader import data_loader, ULE_data_loader
from misc.utils import generate_trajectories


def main(opt):
    print(opt.system)
    print("Train") if opt.train else print("Test")
    opt.M, opt.L, opt.test_L = opt.n_trj, opt.n_step, opt.test_n_step

    # set the input dimension
    if opt.system[:3] == 'OLE':
        opt.input_dim = opt.dim
    elif opt.system[:3] == 'ULE':
        opt.input_dim = 2 * opt.dim
    if opt.time_dependent:
        opt.input_dim += 1

    # Generate trajectories and samples with a time-step \Delta t = opt.time_step 
    # (simulation time-step \delta t = opt.time_step/opt.sample_freq)
    tot_trajs_t, tot_test_trajs_t, drift_force, diff_mat = generate_trajectories(opt)

    # train with varying the random seed
    for seed in range(opt.init_seed, opt.init_seed + opt.trial):        
        opt.seed = seed
        opt.save = opt.save_path + '/seed%d' %opt.seed

        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)

        print('seed, T, dim, sample_freq:', opt.seed, opt.Tc, opt.dim, opt.sample_freq)
        if opt.diff_use_drift:
            print("Diff_use_drift....")

        # Normalize data to be on a similar scale
        args = {}
        
        if opt.system[:3] == 'OLE':
            original_trajs_t = tot_trajs_t[(opt.seed-opt.init_seed) * opt.M : (opt.seed-opt.init_seed) * opt.M + opt.M]
            original_test_trajs_t = tot_test_trajs_t[(opt.seed-opt.init_seed) * opt.M : (opt.seed-opt.init_seed) * opt.M + opt.M]
        
        elif opt.system[:3] == 'ULE':
            original_trajs_t = tot_trajs_t[(opt.seed-opt.init_seed) * opt.M : (opt.seed-opt.init_seed) * opt.M + opt.M, :, :opt.dim]
            original_test_trajs_t = tot_test_trajs_t[(opt.seed-opt.init_seed) * opt.M : (opt.seed-opt.init_seed) * opt.M + opt.M, :, :opt.dim]

        # add measurement_noise
        trajs_t = original_trajs_t + torch.randn_like(original_trajs_t) * np.sqrt(opt.Tc) * opt.m_noise
        test_trajs_t = original_test_trajs_t + torch.randn_like(original_test_trajs_t) * np.sqrt(opt.Tc) * opt.m_noise

        if opt.normalize:
            mean, std = trajs_t.mean(axis=(0, 1)), trajs_t.std(axis=(0, 1))
        else:
            mean, std = torch.zeros(trajs_t.shape[-1]), torch.ones(trajs_t.shape[-1])

        norm_trajs_t = (trajs_t - mean) / std
        norm_test_trajs_t = (test_trajs_t - mean) / std
        print("Normalize: ", mean, std)

        # Split dataset into train, valid, and test datasetes
        if opt.time_dependent == 0:
            train_size = int((1-opt.val_size) * opt.L)

            args['mean'], args['std'] = mean, std
            args['norm_trajs_t'] = norm_trajs_t[:, :train_size]
            args['norm_valid_trajs_t'] = norm_trajs_t[:, train_size:]
            args['norm_test_trajs_t'] = norm_test_trajs_t
        else:
            train_size = int((1-opt.val_size) * opt.M)

            args['mean'], args['std'] = mean, std
            args['norm_trajs_t'] = norm_trajs_t[:train_size]
            args['norm_valid_trajs_t'] = norm_trajs_t[train_size:]
            args['norm_test_trajs_t'] = norm_test_trajs_t

        # set the answer
        if opt.system[:3] == 'OLE':
            args['train_drift_ans'] = drift_force(original_trajs_t[:, :train_size], opt)
            args['valid_drift_ans'] = drift_force(original_trajs_t[:, train_size:], opt)
            args['test_drift_ans'] = drift_force(original_test_trajs_t, opt)
            args['train_diff_ans'] = diff_mat(original_trajs_t[:, :train_size], opt)
            args['valid_diff_ans'] = diff_mat(original_trajs_t[:, train_size:], opt)
            args['test_diff_ans']= diff_mat(original_test_trajs_t, opt)

            delta_norm_trajs_t = torch.diff(norm_trajs_t, axis=-2).reshape(-1, norm_trajs_t.shape[-1])
            delta2_norm_trajs_t = torch.einsum('bi,bj->bij', delta_norm_trajs_t, delta_norm_trajs_t)

        elif opt.system[:3] == 'ULE':
            args['norm_vel_trajs_t'] = (args['norm_trajs_t'][:, 1:] - args['norm_trajs_t'][:, :-1])/(opt.time_step)
            args['norm_valid_vel_trajs_t'] = (args['norm_valid_trajs_t'][:, 1:] - args['norm_valid_trajs_t'][:, :-1])/(opt.time_step)
            args['norm_test_vel_trajs_t'] = (args['norm_test_trajs_t'][:, 1:] - args['norm_test_trajs_t'][:, :-1])/(opt.time_step)

            if opt.time_dependent == 0:
                args['norm_tot_trajs_t'] = torch.cat([args['norm_trajs_t'][:, :-1], args['norm_vel_trajs_t']], dim=-1)
                args['norm_valid_tot_trajs_t'] = torch.cat([args['norm_valid_trajs_t'][:, :-1], args['norm_valid_vel_trajs_t']], dim=-1)
                args['norm_test_tot_trajs_t'] = torch.cat([args['norm_test_trajs_t'][:, :-1], args['norm_test_vel_trajs_t']], dim=-1)

                args['train_drift_ans'] = drift_force(tot_trajs_t[(opt.seed-opt.init_seed) * opt.M : (opt.seed-opt.init_seed) * opt.M + opt.M, :train_size-1], opt)
                args['valid_drift_ans'] = drift_force(tot_trajs_t[(opt.seed-opt.init_seed) * opt.M : (opt.seed-opt.init_seed) * opt.M + opt.M, train_size:-1], opt)
                args['test_drift_ans'] = drift_force(tot_test_trajs_t[(opt.seed-opt.init_seed) * opt.M : (opt.seed-opt.init_seed) * opt.M + opt.M, :-1], opt)
                args['train_diff_ans'] = diff_mat(tot_trajs_t[(opt.seed-opt.init_seed) * opt.M : (opt.seed-opt.init_seed) * opt.M + opt.M, :train_size-1], opt)
                args['valid_diff_ans'] = diff_mat(tot_trajs_t[(opt.seed-opt.init_seed) * opt.M : (opt.seed-opt.init_seed) * opt.M + opt.M, train_size:-1], opt)
                args['test_diff_ans']= diff_mat(tot_test_trajs_t[(opt.seed-opt.init_seed) * opt.M : (opt.seed-opt.init_seed) * opt.M + opt.M, :-1], opt)

            else:
                train_size = int((1-opt.val_size) * opt.M)

                time_data_t = torch.arange(opt.L).unsqueeze(dim=1) * opt.time_step
                time_data_t = (time_data_t - time_data_t.mean())/time_data_t.std()

                args['norm_tot_trajs_t'] = torch.cat([args['norm_trajs_t'][:, :-1], args['norm_vel_trajs_t'], time_data_t[:-1].expand(args['norm_trajs_t'].shape[0], -1, 1)], dim=-1)
                args['norm_valid_tot_trajs_t'] = torch.cat([args['norm_valid_trajs_t'][:, :-1], args['norm_valid_vel_trajs_t'], time_data_t[:-1].expand(args['norm_valid_trajs_t'].shape[0], -1, 1)], dim=-1)
                args['norm_test_tot_trajs_t'] = torch.cat([args['norm_test_trajs_t'][:, :-1], args['norm_test_vel_trajs_t'], time_data_t[:-1].expand(args['norm_test_trajs_t'].shape[0], -1, 1)], dim=-1)

                from toy.time_dependent import ret_protocols

                k_list_t, T_list_t = ret_protocols(opt.L, opt.time_step)

                # Compute answers of drift-force and diff-mat
                args['train_drift_ans'] = drift_force(tot_trajs_t[(opt.seed-opt.init_seed) * opt.M : (opt.seed-opt.init_seed) * opt.M + opt.M][:train_size, :-1], k_list_t[:-1]).unsqueeze(dim=-1)
                args['valid_drift_ans'] = drift_force(tot_trajs_t[(opt.seed-opt.init_seed) * opt.M : (opt.seed-opt.init_seed) * opt.M + opt.M][train_size:, :-1], k_list_t[:-1]).unsqueeze(dim=-1)
                args['test_drift_ans'] = drift_force(tot_test_trajs_t[(opt.seed-opt.init_seed) * opt.M : (opt.seed-opt.init_seed) * opt.M + opt.M][:, :-1], k_list_t[:-1]).unsqueeze(dim=-1)
                args['train_diff_ans'] = diff_mat(T_list_t[:-1]).reshape(-1, 1, 1).expand(train_size, -1, 1, 1)
                args['valid_diff_ans'] = diff_mat(T_list_t[:-1]).reshape(-1, 1, 1).expand(opt.M - train_size, -1, 1, 1)
                args['test_diff_ans']= diff_mat(T_list_t[:-1]).reshape(-1, 1, 1).expand(opt.M, -1, 1, 1)
        
            norm_vel_trajs_t = torch.diff(norm_trajs_t, axis=-2)/opt.time_step
            delta_norm_trajs_t = torch.diff(norm_vel_trajs_t, axis=-2).reshape(-1, norm_vel_trajs_t.shape[-1])
            delta2_norm_trajs_t = torch.einsum('bi,bj->bij', delta_norm_trajs_t, delta_norm_trajs_t)

        print("num_trjs, trj_len, input_dim, trajs_t.type, device: ", trajs_t.shape[0], trajs_t.shape[1], opt.input_dim, trajs_t.type(), trajs_t.device)

        if 'drift' in opt.mode:
            args['output_mean'], args['output_std'] = delta_norm_trajs_t.mean(axis=0)/opt.time_step, delta_norm_trajs_t.std(axis=0)/opt.time_step
        
        elif opt.mode == 'diff':
            tril_ind = torch.tril_indices(opt.dim, opt.dim)

            args['output_mean'], args['output_std'] = delta2_norm_trajs_t.mean(axis=0)/(2*opt.time_step), delta2_norm_trajs_t.std(axis=0)/(2*opt.time_step)
            args['output_mean'], args['output_std'] = scipy.linalg.sqrtm(args['output_mean'])[..., tril_ind[0], tril_ind[1]], scipy.linalg.sqrtm(args['output_std'])[..., tril_ind[0], tril_ind[1]]
            args['output_mean'], args['output_std'] = torch.tensor(args['output_mean']), torch.tensor(args['output_std'])

        opt.output_mean, opt.output_std = args['output_mean'], args['output_std']
        print("Output Normalize: ", opt.output_mean, opt.output_std)

        # Data loader
        opt.n_batch = int(np.ceil(train_size/opt.batch_size))
        if opt.system[:3] == 'OLE':
            train_loader, valid_loader, test_loader = data_loader(
                args['norm_trajs_t'], 
                args['norm_valid_trajs_t'], 
                args['norm_test_trajs_t'],
                opt
            )
        elif opt.system[:3] == 'ULE':
            train_loader, valid_loader, test_loader = ULE_data_loader(
                args['norm_tot_trajs_t'], 
                args['norm_valid_tot_trajs_t'],
                args['norm_test_tot_trajs_t'], 
                opt
            )

        # Make and train the model called LBN!
        if 'drift' in opt.mode:
            print("LBN_%s, n_layer, n_blayer, n_hidden, lr " %opt.mode, opt.n_layer, opt.n_blayer, opt.n_hidden, opt.lr_drift)
            if opt.BNN_mode == 'BNN':
                model = LBNN_Drift(opt)
            else:
                model = LNN_Drift(opt)

            model = model.to(opt.device)
            optim = torch.optim.Adam(model.force_model.parameters(), lr=opt.lr_drift)

        elif opt.mode == 'diff':
            print("LBN_%s, n_layer, n_blayer, n_hidden, lr " %opt.mode, opt.n_layer, opt.n_blayer, opt.n_hidden, opt.lr_diff)
            print(opt.diff_mode)
            if opt.BNN_mode == 'BNN':
                model = LBNN_Diff(opt)
            else:
                model = LNN_Diff(opt)

            model = model.to(opt.device)
            if opt.diff_mode == 'homogeneous':
                if opt.BNN_mode == 'BNN':
                    optim = torch.optim.Adam(model.diff_model.parameters(), lr=opt.lr_diff)
                else:
                    optim = torch.optim.Adam([model.tril_param], lr=opt.lr_diff)

            elif opt.diff_mode == 'inhomogeneous':
                optim = torch.optim.Adam(model.diff_model.parameters(), lr=opt.lr_diff)

        opt.best_loss = np.inf
        opt.best_nmse = np.inf
        if opt.load == 1:
            checkpoint = torch.load(opt.save + '/checkpoint_%s.pth.tar' %opt.mode)
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['optimizer'])

            opt.best_loss = checkpoint['best_loss']
            opt.best_nmse = checkpoint['best_nmse']

        # train the LBN...!
        if opt.train:
            ret_train, ret_valid = train(model, train_loader, valid_loader, optim, opt, args)

        # evaluate on test set
        ret_test = evaluate(test_loader, opt, args)

        # Save logs and hyperparameters
        if opt.train:
            train_df = pd.DataFrame(ret_train)
            valid_df = pd.DataFrame(ret_valid)
        test_df = pd.DataFrame(ret_test)

        if opt.train:
            if opt.diff_use_drift and opt.mode=='diff':
                train_df.to_csv(os.path.join(opt.save, "diff_udT_train_log.csv"), index=False)
                valid_df.to_csv(os.path.join(opt.save, "diff_udT_valid_log.csv"), index=False)

            else:
                train_df.to_csv(os.path.join(opt.save, "%s_train_log.csv" %opt.mode), index=False)
                valid_df.to_csv(os.path.join(opt.save, "%s_valid_log.csv" %opt.mode), index=False)
        
        if opt.diff_use_drift and opt.mode=='diff':
            test_df.to_csv(os.path.join(opt.save, "diff_udT_test_log.csv"), index=False)
        else:
            test_df.to_csv(os.path.join(opt.save, "%s_test_log.csv" %opt.mode), index=False)

        opt_save = copy.deepcopy(opt)

        opt_save.device = "cuda" if use_cuda else "cpu"
        opt_save.mean = mean.numpy().astype(np.float64).tolist()
        opt_save.std = std.numpy().astype(np.float64).tolist()
        opt_save.output_mean = opt.output_mean.numpy().astype(np.float64).tolist()
        opt_save.output_std = opt.output_std.numpy().astype(np.float64).tolist()

        hparams = json.dumps(vars(opt_save))
        with open(os.path.join(opt.save, "hparams_%s.json" %opt.mode), "w") as f:
            f.write(hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Langevin Bayesian Neural networks (LBN)"
    )
    parser.add_argument(
        "--system",
        default=None,
        type=str,
        metavar="SYS",
        help="Select a Langevin system we want to infer (default: none): OLE_Nonliear, OLE_Inhomogeneous, OLE_HHmodel, ULE_Nonlinear, ULE_BrownianCarnot.",
    )
    parser.add_argument(
        "--data",
        default=None,
        type=str,
        metavar="DATA",
        help="Data load",
    )
    parser.add_argument(
        "--train", 
        type=int, 
        default=1, 
        help="Train or not (default: 1)"
    )
    parser.add_argument(
        "--normalize",
        default=1,
        type=int,
        help="Normalize (default: 1)",
    )
    parser.add_argument(
        "--Tc",
        type=float,
        default=1,
        metavar="T",
        help="Heat bath temperature (default: 1)",
    )
    parser.add_argument(
        "--init-seed",
        "-IS",
        type=int,
        default=1,
        metavar="M",
        help="Initial seed (default: 1)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        metavar="d",
        help="number of coordinates (default: 2)",
    )
    parser.add_argument(
        "--n-trj",
        "-M",
        type=int,
        default=1,
        metavar="M",
        help="number of trajectories (default: 1)",
    )
    parser.add_argument(
        "--n-step",
        "-L",
        type=int,
        default=10000,
        metavar="L",
        help="number of step for each trajectory (default: 10000)",
    )
    parser.add_argument(
        "--test-n-step",
        "-TL",
        type=int,
        default=10000,
        metavar="L",
        help="number of step for each trajectory (default: 10000)",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=1e-2,
        help="time step size of simulation (default: 0.01)",
    )
    parser.add_argument(
        "--save-path",
        default="./checkpoint",
        type=str,
        metavar="PATH",
        help="path to save result (default: none)",
    )
    parser.add_argument(
        "--load",
        default=0,
        type=int,
        metavar="LOAD",
        help="load (default: 0)",
    )
    parser.add_argument(
        "--val-size",
        "-V",
        type=float,
        default=0.2,
        metavar="V",
        help="The relative size of validation set (default: 0.2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--lr-drift",
        type=float,
        default=1e-6,
        metavar="LR",
        help="learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--lr-diff",
        type=float,
        default=5e-6,
        metavar="LR",
        help="learning rate (default: 5e-6)",
    )
    parser.add_argument(
        "--kl-w",
        type=float,
        default=0.01,
        metavar="KLWD",
        help="Model complexity weight (default: 1e-2)",
    )
    parser.add_argument(
        "--n-iter-drift",
        type=int,
        default=5000,
        metavar="N",
        help="number of iteration to train (default: 5000)",
    )
    parser.add_argument(
        "--n-iter-diff",
        type=int,
        default=10000,
        metavar="N",
        help="number of iteration to train (default: 10000)",
    )
    parser.add_argument(
        "--sample-freq",
        type=int,
        default=10,
        metavar="SF",
        help="sampling frequency (default: 10)",
    )
    parser.add_argument(
        "--n-layer",
        type=int,
        default=4,
        metavar="N",
        help="number of layers (default: 4)",
    )
    parser.add_argument(
        "--n-blayer",
        type=int,
        default=2,
        metavar="BN",
        help="number of stochastic layers (default: 2)",
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=256,
        metavar="N",
        help="number of hidden neuron (default: 256)",
    )
    parser.add_argument(
        "--no-cuda", 
        action="store_true", 
        default=False, 
        help="disables CUDA training"
    )
    parser.add_argument(
        "--trial", 
        type=int, 
        default=1, 
        metavar="S", 
        help="number of trials (default: 1)"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default='drift', 
        metavar="MD", 
        help="train mode (default: drift): 'drift' or 'diff' "
    )
    parser.add_argument(
        "--diff-mode", 
        type=str, 
        default='homogeneous', 
        metavar="DMD", 
        help="diffusion mode (default: homogeneous): 'homogeneous' or 'inhomogeneous' "
    )
    parser.add_argument(
        "--time-dependent", 
        type=int, 
        default=0, 
        help="Time dependent or not (default: 0)"
    )
    parser.add_argument(
        "--BNN-mode", 
        type=str, 
        default='BNN', 
        metavar="NET", 
        help="BNN mode (default: BNN): 'DNN' or 'BNN' "
    )
    parser.add_argument(
        "--n-ensemble", 
        type=int, 
        default=200, 
        metavar="N", 
        help="number of ensemble for output (default: 200)"
    )
    parser.add_argument(
        "--n-mc", 
        type=int, 
        default=3, 
        metavar="N", 
        help="number of the weights to gather the loss via Monte-Carlo method"
    )
    parser.add_argument(
        "--diff-use-drift", 
        type=int, 
        default=0, 
        help="use the drift field to infer the diffusion matrix (default: 0)"
    )
    parser.add_argument(
        "--alpha", 
        type=int, 
        default=10, 
        metavar="N", 
        help="parameters for nonlinear forces"
    )
    parser.add_argument(
        "--m-noise", 
        type=float, 
        default=0.0, 
        metavar="MNoise", 
        help="measurement noise (default: 0)"
    )

    opt = parser.parse_args()
    use_cuda = not opt.no_cuda and torch.cuda.is_available()

    opt.device = torch.device("cuda" if use_cuda else "cpu")
    print(opt.device)
    
    main(opt)