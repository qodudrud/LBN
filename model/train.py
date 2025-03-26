import os
import json
import copy

import numpy as np
import torch

from model.net import LNN_Drift, LNN_Diff, LBNN_Drift, LBNN_Diff
from model.bnn_loss import *

from misc.utils import AverageMeter, logging, logging_real, logging_test, logging_total_test, logging_total_test_real, save_checkpoint, save_checkpoint_real

def _load_drift(opt):
    opt_drift = copy.deepcopy(opt)
    with open(opt.save + '/hparams_drift.json', 'r') as f:
        hparams_drift = json.load(f)

    opt_drift.output_mean = torch.tensor(hparams_drift['output_mean'])
    opt_drift.output_std = torch.tensor(hparams_drift['output_std'])

    if opt.BNN_mode == 'BNN':
        drift_model = LBNN_Drift(opt_drift)
    else:
        drift_model = LNN_Drift(opt_drift)

    drift_model = drift_model.to(opt.device)

    checkpoint = torch.load(opt.save + '/model_best_drift.pth.tar')
    drift_model.load_state_dict(checkpoint['state_dict'])

    # print("Diff_use_drift....")
    # print(checkpoint['epoch'], checkpoint['best_loss'], checkpoint['best_nmse'])
    return drift_model, opt_drift


def _train(opt, model, optim, train_trajs_t, train_loader, std = None):
    if std is None:
        std = torch.ones(train_trajs_t.size(-1))

    # The KL-divergence between posterior and prior distributions
    kl_loss = Bayesian_KL_Loss()

    train_losses = AverageMeter("TrainLoss")
    train_KLlosses = AverageMeter("TrainKLD")

    if opt.mode == 'diff' and opt.diff_use_drift:
        drift_model, opt_drift = _load_drift(opt)

    model.train()
    for data, labels in train_loader:
        kl = 0
        train_loss = 0

        for _ in range(opt.n_mc):
            loc = model(data.to(opt.device))

            if 'drift' in opt.mode:
                dist = torch.distributions.MultivariateNormal(loc=loc, scale_tril = torch.eye(opt.dim).to(opt.device))
            else:
                if opt.diff_use_drift:
                    drift = drift_model(data.to(opt.device))
                    drift2 = torch.einsum('...i,...j->...ij', drift, drift)
                    drift2 = drift2[..., model.tril_ind[0], model.tril_ind[1]]

                    dist = torch.distributions.MultivariateNormal(loc = 2 * loc + drift2, scale_tril = torch.eye(loc.shape[-1]).to(opt.device))

                else:
                    dist = torch.distributions.MultivariateNormal(loc = 2 * loc, scale_tril = torch.eye(loc.shape[-1]).to(opt.device))

            train_loss += -dist.log_prob(labels.to(opt.device)).mean()

        train_loss /= opt.n_mc

        kl = kl_loss(model).item()
        train_loss += opt.kl_w/opt.n_batch * kl
        
        optim.zero_grad()
        
        train_loss.backward()
        optim.step()
        
        train_KLlosses.update(kl, 1)
        train_losses.update(train_loss.item(), data.shape[0])

    if 'drift' in opt.mode:
        train_model = model.force_model(train_trajs_t.to(opt.device)) * std.to(opt.device)
    else:
        train_model = model.diff_mat(train_trajs_t.to(opt.device)) * torch.outer(std, std).to(opt.device)
        if opt.system[:3] == 'ULE' and opt.sample_freq > 1:
            train_model *= 3/2

    return train_model, train_losses, train_KLlosses


def _validate(opt, model, valid_trajs_t, valid_loader, std = None):
    if std is None:
        std = torch.ones(valid_trajs_t.size(-1))

    valid_losses = AverageMeter("ValidLoss")

    if opt.mode == 'diff' and opt.diff_use_drift:
        drift_model, opt_drift = _load_drift(opt)

    model.eval()
    with torch.no_grad():
        for data, labels in valid_loader:
            valid_loss = 0

            for _ in range(opt.n_mc):
                loc = model(data.to(opt.device))
                if 'drift' in opt.mode:
                    dist = torch.distributions.MultivariateNormal(loc=loc, scale_tril = torch.eye(opt.dim).to(opt.device))

                elif opt.mode == 'diff':
                    if opt.diff_use_drift:
                        drift = drift_model(data.to(opt.device))
                        drift2 = torch.einsum('...i,...j->...ij', drift, drift)
                        drift2 = drift2[..., model.tril_ind[0], model.tril_ind[1]]

                        dist = torch.distributions.MultivariateNormal(loc = 2 * loc + drift2, scale_tril = torch.eye(loc.shape[-1]).to(opt.device))

                    else:
                        dist = torch.distributions.MultivariateNormal(loc = 2 * loc, scale_tril = torch.eye(loc.shape[-1]).to(opt.device))

                valid_loss += -dist.log_prob(labels.to(opt.device)).mean()

            valid_loss /= opt.n_mc

            valid_losses.update(valid_loss.item(), data.shape[0])

        if 'drift' in opt.mode:
            valid_model = model.force_model(valid_trajs_t.to(opt.device)) * std.to(opt.device)
        else:
            valid_model = model.diff_mat(valid_trajs_t.to(opt.device)) * torch.outer(std, std).to(opt.device)
            if opt.system[:3] == 'ULE' and opt.sample_freq > 1:
                valid_model *= 3/2
    
    return valid_model, valid_losses


def train(model, train_loader, valid_loader, optim, opt, args):
    std = args['std']

    if opt.system[:3] == 'OLE':
        norm_trajs_t = args['norm_trajs_t']
        norm_valid_trajs_t = args['norm_valid_trajs_t']
    elif opt.system[:3] == 'ULE':
        norm_trajs_t = args['norm_tot_trajs_t']
        norm_valid_trajs_t = args['norm_valid_tot_trajs_t']

    if 'drift' in opt.mode:
        train_ans = args['train_drift_ans']
        valid_ans = args['valid_drift_ans']
    else:
        train_ans = args['train_diff_ans']
        valid_ans = args['valid_diff_ans']
    
    # For saving results...
    ret_train = []
    ret_valid = []
    
    # makedirs for saving the results...
    try:
        if not os.path.exists(opt.save):
            os.makedirs(opt.save)
    except:
        if os.path.exists(opt.save):
            print("makedirs are already performed...")
        else:
            print("makedirs are failed....")

    # Let's train the model
    best_valid_loss = opt.best_loss
    best_valid_nmse = opt.best_nmse

    n_iter = opt.n_iter_drift if 'drift' in opt.mode else opt.n_iter_diff
    if hasattr(opt, 'optimizer'):
        if opt.optimizer == 'SGD':
            print("SGD, CosineAnnealingLR")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_iter, eta_min=0)

    for epoch in range(1, n_iter + 1):
        train_model, train_losses, train_KLlosses = _train(opt, model, optim, norm_trajs_t, train_loader, std=std)
        train_log = logging(epoch, train_losses, train_ans.to(opt.device), train_model.to(opt.device), KLD=train_KLlosses, mode=opt.mode)

        valid_model, valid_losses = _validate(opt, model, norm_valid_trajs_t, valid_loader, std=std)
        valid_log = logging(epoch, valid_losses, valid_ans.to(opt.device), valid_model.to(opt.device), train=False, mode=opt.mode)

        is_best = valid_losses.avg < best_valid_loss
        if is_best:
            best_valid_loss = valid_log["loss"]
            best_valid_nmse = valid_log["%s_nmse" %opt.mode]

        valid_log['best_loss'] = best_valid_loss
        valid_log['best_nmse'] = best_valid_nmse

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_loss": best_valid_loss,
                "best_nmse": best_valid_nmse,
                "optimizer": optim.state_dict(),
            },
            is_best,
            opt.save,
            mode=opt.mode
        )

        ret_train.append(train_log)
        ret_valid.append(valid_log)

        if hasattr(opt, 'optimizer'):
            if opt.optimizer == 'SGD':
                scheduler.step()

    # save_checkpoint(
    #     {
    #         "epoch": epoch,
    #         "state_dict": model.state_dict(),
    #         "best_loss": best_valid_loss,
    #         "best_nmse": best_valid_nmse,
    #         "optimizer": optim.state_dict(),
    #     },
    #     is_best,
    #     opt.save,
    #     mode=opt.mode
    # )

    return ret_train, ret_valid


def train_real(model, train_loader, valid_loader, optim, opt, args):
    std = args['std']

    if opt.system[:3] == 'OLE':
        norm_trajs_t = args['norm_trajs_t']
        norm_valid_trajs_t = args['norm_valid_trajs_t']
    elif opt.system[:3] == 'ULE':
        norm_trajs_t = args['norm_tot_trajs_t']
        norm_valid_trajs_t = args['norm_valid_tot_trajs_t']
    
    # For saving results...
    ret_train = []
    ret_valid = []

    try:
        if not os.path.exists(opt.save):
            os.makedirs(opt.save)
    except:
        if os.path.exists(opt.save):
            print("makedirs are already performed...")
        else:
            print("makedirs are failed....")

    # Let's train the model
    best_valid_loss = np.inf

    n_iter = opt.n_iter_drift if 'drift' in opt.mode else opt.n_iter_diff
    for epoch in range(1, n_iter + 1):
        train_model, train_losses, train_KLlosses = _train(opt, model, optim, norm_trajs_t, train_loader, std=std)
        train_log = logging_real(epoch, train_losses, KLD=train_KLlosses, mode=opt.mode)

        valid_model, valid_losses = _validate(opt, model, norm_valid_trajs_t, valid_loader, std=std)
        valid_log = logging_real(epoch, valid_losses, train=False, mode=opt.mode)

        is_best = valid_losses.avg < best_valid_loss
        if is_best:
            best_valid_loss = valid_log["loss"]

        valid_log['best_loss'] = best_valid_loss

        if epoch % 10 == 0:
            save_checkpoint_real(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_loss": best_valid_loss,
                    "optimizer": optim.state_dict(),
                },
                is_best,
                opt.save,
                mode=opt.mode,
                ihm = True if opt.diff_mode == 'inhomogeneous' else False
            )

        ret_train.append(train_log)
        ret_valid.append(valid_log)

    return ret_train, ret_valid



def evaluate(test_loader, opt, args):
    std = args['std']

    if opt.system[:3] == 'OLE':
        norm_test_trajs_t = args['norm_test_trajs_t']
    elif opt.system[:3] == 'ULE':
        norm_test_trajs_t = args['norm_test_tot_trajs_t']
    
    if 'drift' in opt.mode:
        test_ans = args['test_drift_ans']
    else:
        test_ans = args['test_diff_ans']

    # Load the best model
    if 'drift' in opt.mode:
        if opt.BNN_mode == 'BNN':
            model = LBNN_Drift(opt)
        else:
            model = LNN_Drift(opt)

    elif opt.mode == 'diff':
        if opt.BNN_mode == 'BNN':
            model = LBNN_Diff(opt)
        else:
            model = LNN_Diff(opt)

    if opt.mode == 'drift_all':
        model = model.to(opt.device)
        checkpoint = torch.load(opt.save + '/model_best_drift.pth.tar')

        model.load_state_dict(checkpoint['state_dict'])
        print(checkpoint['epoch'], checkpoint['best_loss'], checkpoint['best_nmse'])

        model_r = LBNN_Drift(opt)
        model_r.to(opt.device)

        checkpoint_r = torch.load(opt.save + '/model_best_drift_r.pth.tar')
        model_r.load_state_dict(checkpoint_r['state_dict'])
        print(checkpoint_r['epoch'], checkpoint_r['best_loss'], checkpoint_r['best_nmse'])

    else:
        model = model.to(opt.device)
        checkpoint = torch.load(opt.save + '/model_best_%s.pth.tar' %opt.mode)

        model.load_state_dict(checkpoint['state_dict'])
        print(checkpoint['epoch'], checkpoint['best_loss'], checkpoint['best_nmse'])


    if opt.mode == 'diff' and opt.diff_use_drift:
        drift_model, opt_drift = _load_drift(opt)
    
    # For saving results...
    ret_test = []
    test_losses = AverageMeter("TestLoss")

    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            loc = model(data.to(opt.device))

            if 'drift' in opt.mode:
                dist = torch.distributions.MultivariateNormal(loc=loc, scale_tril = torch.eye(opt.dim).to(opt.device))

            elif opt.mode == 'diff':
                if opt.diff_use_drift:
                    drift = drift_model(data.to(opt.device))
                    drift2 = torch.einsum('...i,...j->...ij', drift, drift)
                    drift2 = drift2[..., model.tril_ind[0], model.tril_ind[1]]

                    dist = torch.distributions.MultivariateNormal(loc = 2 * loc + drift2, scale_tril = torch.eye(loc.shape[-1]).to(opt.device))

                else:
                    dist = torch.distributions.MultivariateNormal(loc = 2 * loc, scale_tril = torch.eye(loc.shape[-1]).to(opt.device))
            
            test_loss = -dist.log_prob(labels.to(opt.device)).mean()
            test_losses.update(test_loss.item(), data.shape[0])

        test_models = []
        for _ in range(opt.n_ensemble):
            if 'drift' in opt.mode:
                test_model = model.force_model(norm_test_trajs_t.to(opt.device)) * std.to(opt.device)
                
                if opt.mode == 'drift_all':
                    test_model_r = model_r.force_model(norm_test_trajs_t.to(opt.device)) * std.to(opt.device)

                    test_model = test_model + (test_model - test_model_r)/4

            elif opt.mode == 'diff':
                test_model = model.diff_mat(norm_test_trajs_t.to(opt.device)) * torch.outer(std, std).to(opt.device)
                if opt.system[:3] == 'ULE' and opt.sample_freq > 1:
                    test_model *= 3/2

            test_models.append(test_model.cpu().detach().numpy())

            test_log = logging_test(checkpoint['epoch'], test_losses, test_ans.to(opt.device), test_model.to(opt.device), mode=opt.mode)

            ret_test.append(test_log)
        
    # preds = [log['%s_rmse' %opt.mode] for log in ret_test]
    # preds = np.stack(preds)
    # means = preds.mean(axis=0)
    # stds = preds.std(axis=0)

    # print("Mean RMSE: %.4f, Stdv. RMSE: %.4f" %(means, stds))

    # evaluate total test error and uncertainty...!!
    test_models = torch.Tensor(np.array(test_models))
    if opt.mode == 'diff':
        print(torch.mean(test_models, axis=0))

    total_test_log = logging_total_test(test_ans, test_models, mode=opt.mode)
    ret_test.append(total_test_log)

    return ret_test


def evaluate_real(test_loader, opt, args):
    std = args['std']

    if opt.system[:3] == 'OLE':
        norm_test_trajs_t = args['norm_valid_trajs_t']
    elif opt.system[:3] == 'ULE':
        norm_test_trajs_t = args['norm_valid_trajs_t']

    # Load the best model
    if 'drift' in opt.mode:
        if opt.BNN_mode == 'BNN':
            model = LBNN_Drift(opt)
        else:
            model = LNN_Drift(opt)

    elif opt.mode == 'diff':
        if opt.BNN_mode == 'BNN':
            model = LBNN_Diff(opt)
        else:
            model = LNN_Diff(opt)

    model = model.to(opt.device)

    if opt.mode == 'diff' and opt.diff_mode == 'inhomogeneous':
        checkpoint = torch.load(opt.save + '/model_best_diff_ihm.pth.tar')

    else:
        checkpoint = torch.load(opt.save + '/model_best_%s.pth.tar' %opt.mode)

    model.load_state_dict(checkpoint['state_dict'])
    print(checkpoint['epoch'], checkpoint['best_loss'])

    if opt.mode == 'diff' and opt.diff_use_drift:
        drift_model, opt_drift = _load_drift(opt)
    
    # For saving results...
    ret_test = []
    test_losses = AverageMeter("TestLoss")

    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            loc = model(data.to(opt.device))

            if 'drift' in opt.mode:
                dist = torch.distributions.MultivariateNormal(loc=loc, scale_tril = torch.eye(opt.dim).to(opt.device))

            elif opt.mode == 'diff':
                if opt.diff_use_drift:
                    drift = drift_model(data.to(opt.device))
                    drift2 = torch.einsum('...i,...j->...ij', drift, drift)
                    drift2 = drift2[..., model.tril_ind[0], model.tril_ind[1]]

                    dist = torch.distributions.MultivariateNormal(loc = 2 * loc + drift2, scale_tril = torch.eye(loc.shape[-1]).to(opt.device))

                else:
                    dist = torch.distributions.MultivariateNormal(loc = 2 * loc, scale_tril = torch.eye(loc.shape[-1]).to(opt.device))
            
            test_loss = -dist.log_prob(labels.to(opt.device)).mean()
            test_losses.update(test_loss.item(), data.shape[0])

        test_models = []
        for _ in range(opt.n_ensemble):
            if 'drift' in opt.mode:
                test_model = model.force_model(norm_test_trajs_t.to(opt.device)) * std.to(opt.device) 
            elif opt.mode == 'diff':
                test_model = model.diff_mat(norm_test_trajs_t.to(opt.device)) * torch.outer(std, std).to(opt.device)
                if opt.system[:3] == 'ULE' and opt.sample_freq > 1:
                    test_model *= 3/2

            test_models.append(test_model.cpu().detach().numpy())

            test_log = logging_real(checkpoint['epoch'], test_losses, train=False, mode=opt.mode)
            ret_test.append(test_log)
        
    # evaluate total test error and uncertainty...!!
    test_models = np.array(test_models)
    total_test_log = logging_total_test_real(test_models, mode=opt.mode)

    # total_test_log = logging_total_test(np.array(test_ans), test_models, mode=opt.mode)
    ret_test.append(total_test_log)

    return ret_test



