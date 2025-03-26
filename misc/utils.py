import math
import copy
import distutils
import shutil
import random
import gc

import os.path as path
from collections import OrderedDict

import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

def generate_trajectories(opt):
    """
    Generate trajectories for the given system.
    """  
    tot_trajs_t, tot_test_trajs_t, drift_force, diff_mat = None, None, None, None

    if opt.system == 'OLE_Nonlinear':
        from toy.nonlinear import drift_force, diff_mat, simulation_nonlinear

        tot_trajs_t = simulation_nonlinear(opt.trial * opt.M, opt.L, opt.dim, opt.Tc, opt.time_step/opt.sample_freq, sample_freq = opt.sample_freq, seed=0, alpha=opt.alpha).float()
        tot_test_trajs_t = simulation_nonlinear(opt.trial * opt.M, opt.test_L, opt.dim, opt.Tc, opt.time_step/opt.sample_freq, sample_freq = opt.sample_freq, seed=1).float()

    elif opt.system == 'OLE_Inhomogeneous':
        from toy.inhomogeneous import drift_force, diff_mat, simulation_inhomogeneous

        tot_trajs_t = simulation_inhomogeneous(opt.trial * opt.M, opt.L, opt.dim, opt.Tc, opt.time_step/opt.sample_freq, sample_freq = opt.sample_freq, seed=0).float()
        tot_test_trajs_t = simulation_inhomogeneous(opt.trial * opt.M, opt.test_L, opt.dim, opt.Tc, opt.time_step/opt.sample_freq, sample_freq = opt.sample_freq, seed=1).float()

    elif opt.system == 'OLE_HHmodel':
        from toy.HH_model import drift_force, diff_mat, simulation_HHmodel

        tot_trajs_t = simulation_HHmodel(opt.trial * opt.M, opt.L, opt.Tc, opt.time_step/opt.sample_freq, sample_freq = opt.sample_freq, seed=0).float()
        tot_test_trajs_t = simulation_HHmodel(opt.trial * opt.M, opt.test_L, opt.Tc, opt.time_step/opt.sample_freq, sample_freq = opt.sample_freq, seed=1).float()

    elif opt.system == 'OLE_StochasticHHmodel':
        from toy.Stochastic_HH_model import drift_force, diff_mat, simulation_HHmodel

        tot_trajs_t = simulation_HHmodel(opt.trial * opt.M, opt.L, opt.Tc, opt.time_step/opt.sample_freq, sample_freq = opt.sample_freq, seed=0).float()
        tot_test_trajs_t = simulation_HHmodel(opt.trial * opt.M, opt.test_L, opt.Tc, opt.time_step/opt.sample_freq, sample_freq = opt.sample_freq, seed=1).float()

    elif opt.system == 'ULE_VdP':
        from toy.ULE_VdP import drift_force, diff_mat, simulation_ULE_VdP

        tot_trajs_t = simulation_ULE_VdP(opt.trial * opt.M, opt.L, opt.dim, opt.Tc, opt.time_step/opt.sample_freq, sample_freq = opt.sample_freq, seed=0).float()
        tot_test_trajs_t = simulation_ULE_VdP(opt.trial * opt.M, opt.test_L, opt.dim, opt.Tc, opt.time_step/opt.sample_freq, sample_freq = opt.sample_freq, seed=1).float()

    elif opt.system == 'ULE_VdP_IHD':
        from toy.ULE_VdP_IHD import drift_force, diff_mat, simulation_ULE_VdP

        tot_trajs_t = simulation_ULE_VdP(opt.trial * opt.M, opt.L, opt.dim, opt.Tc, opt.time_step/opt.sample_freq, sample_freq = opt.sample_freq, seed=0).float()
        tot_test_trajs_t = simulation_ULE_VdP(opt.trial * opt.M, opt.test_L, opt.dim, opt.Tc, opt.time_step/opt.sample_freq, sample_freq = opt.sample_freq, seed=1).float()

    elif opt.system == 'ULE_BCE':
        from toy.time_dependent import simulation_BrownianCarnotEngine, diff_mat, drift_force

        tot_trajs_t = simulation_BrownianCarnotEngine(opt.trial * opt.M, opt.L, opt.time_step/opt.sample_freq, sample_freq = opt.sample_freq, seed=0).float()
        tot_test_trajs_t = simulation_BrownianCarnotEngine(opt.trial * opt.M, opt.L, opt.time_step/opt.sample_freq, sample_freq = opt.sample_freq, seed=1).float()

    if torch.any(torch.isnan(tot_trajs_t)) or torch.any(torch.isnan(tot_test_trajs_t)):
        print(torch.any(torch.isnan(tot_trajs_t)), torch.any(torch.isnan(tot_test_trajs_t)))
        print(torch.isnan(tot_trajs_t[..., 0]).nonzero()[:, 0])
        raise Exception("nan error")

    return tot_trajs_t, tot_test_trajs_t, drift_force, diff_mat


# Data type conversion
def DCN(x):
    return x.data.cpu().numpy()
def CN(x):
    return x.cpu().numpy()
def TTC(x):
    return torch.Tensor(x).cuda()

# Utility classes
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def calc_MSE(field1, field2):
    return ((field1 - field2)**2).mean()


def calc_NMSE(field1, field2):
    return ((field1 - field2)**2).mean()/(field1**2 + field2**2).mean()


def calc_RMSE(true_field, est_field):
    return ((true_field - est_field)**2).mean()/(est_field**2).mean()


def calc_RMSE_points(true_field, est_fields):
    return ((true_field - est_fields)**2).mean(axis=(0))/(est_fields**2).mean(axis=(0))


def calc_uncertainty(est_fields):
    return est_fields.var(axis=0)/(est_fields**2).mean(axis=0)


def r2_loss(true_field, est_field):
    true_field_mean = true_field.squeeze().mean(axis=0)

    ss_tot = ((true_field - true_field_mean) ** 2).sum(axis=-1).mean()
    ss_res = ((true_field - est_field) ** 2).sum(axis=-1).mean()
    r2 = 1 - ss_res / ss_tot
    return r2


def calc_R2(true_field, est_field):
    mode = 'drift' if true_field.shape[-1] != true_field.shape[-2] else 'diff'
    if mode == 'drift':
        true_field = true_field.reshape(-1, true_field.shape[-1])
        est_field = est_field.reshape(-1, est_field.shape[-1])

    elif mode == 'diff':
        # print(true_field, est_field)
        true_field = true_field.reshape(-1, true_field.shape[-2] * true_field.shape[-1])
        est_field = est_field.reshape(-1, est_field.shape[-2] * est_field.shape[-1])

    return r2_loss(true_field, est_field)


def calc_R2_forSFI(true_field, est_field):
    mode = 'drift' if true_field.shape[-1] != true_field.shape[-2] else 'diff'
    if mode == 'drift':
        true_field = true_field.reshape(-1, true_field.shape[-1])
        est_field = est_field.reshape(-1, est_field.shape[-1])

    elif mode == 'diff':
        true_field = true_field.reshape(-1, true_field.shape[-2] * true_field.shape[-1])
        est_field = est_field.reshape(-1, est_field.shape[-2] * est_field.shape[-1])
        
        if np.all(true_field[0] == true_field[2]):
            true_field = true_field.mean(axis=0)
            est_field = est_field.mean(axis=0)

    return r2_loss(true_field, est_field)


def compute_calibration(RMSE, RMV, num_bins=15):
    assert(len(RMV) == len(RMSE))
    assert(num_bins > 0)

    max_rmv = np.quantile(RMV, 0.25)
    min_rmv = np.quantile(RMV, 0.75)
    
    bin_size = (max_rmv-min_rmv) / num_bins
    bins = np.linspace(min_rmv, max_rmv, num_bins + 1)
    indices = np.digitize(RMV, bins, right=True)

    bin_RMSE = np.zeros(num_bins, dtype=float)
    bin_RMV = np.zeros(num_bins, dtype=float)
    bin_counts = np.zeros(num_bins, dtype=int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_RMSE[b] = np.mean(RMSE[selected])
            bin_RMV[b] = np.mean(RMV[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_RMSE * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_RMV * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_RMSE - bin_RMV)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    return { "accuracies": bin_RMSE, 
             "confidences": bin_RMV, 
             "counts": bin_counts, 
             "bins": bins,
             "avg_accuracy": avg_acc,
             "avg_confidence": avg_conf,
             "expected_calibration_error": ece,
             "max_calibration_error": mce }


def logging(i, loss, true_values, estimates, train=True, KLD = None, mode='drift'):
    nmse = calc_NMSE(true_values, estimates).item()
    rmse = calc_RMSE(true_values, estimates).item()
    r2 = calc_R2(true_values, estimates).item()

    tmp = {}

    tmp["epoch"] = i
    tmp["loss"] = loss.avg
    tmp["%s_nmse" %mode] = nmse
    tmp["%s_rmse" %mode] = rmse
    tmp["%s_r2" %mode] = r2
    if train:
        if KLD is None:
            print("Train  epoch: %d  loss: %.5f  %s_nmse: %.5f  rmse: %.5f  r2: %.5f" % (i, loss.avg, mode, nmse, rmse, r2))

        else:
            tmp["%s_KLD" %mode] = KLD.avg
            print("Train  epoch: %d  loss: %.5f  KLD: %.5f  %s_nmse: %.5f  rmse: %.5f  r2: %.5f" % (i, loss.avg, KLD.avg, mode, nmse, rmse, r2))
    else:
        if KLD is None:
            print("Test   epoch: %d  loss: %.5f  %s_nmse: %.5f  rmse: %.5f  r2: %.5f" % (i, loss.avg, mode, nmse, rmse, r2))

        else:
            tmp["%s_KLD" %mode] = KLD.avg
            print("Test  epoch: %d  loss: %.5f  KLD: %.5f  %s_nmse: %.5f  rmse: %.5f  r2: %.5f" % (i, loss.avg, KLD.avg, mode, nmse, rmse, r2))

    return tmp


def logging_test(i, loss, true_values, estimates, KLD = None, mode='drift'):
    mse = calc_MSE(true_values, estimates).item()
    y2_mean = (estimates**2).mean()
    r2 = calc_R2(true_values, estimates).item()

    tmp = {}

    tmp["epoch"] = i
    tmp["loss"] = loss.avg
    tmp["%s_mse" %mode] = mse
    tmp["%s^2_mean" %mode] = y2_mean.item()
    tmp["%s_r2" %mode] = r2

    if KLD is None:
        print("Test   epoch: %d  loss: %.5f  %s_mse: %.5f  y2: %.5f  r2: %.5f" 
              % (i, loss.avg, mode, mse, y2_mean, r2))

    else:
        tmp["%s_KLD" %mode] = KLD.avg
        print("Test  epoch: %d  loss: %.5f  KLD: %.5f  %s_mse: %.5f  y2: %.5f  r2: %.5f" 
              % (i, loss.avg, KLD.avg, mode, mse, y2_mean, r2))

    return tmp


def logging_total_test(true_values, ensemble_estimates, mode='drift'):
    mean_estimates = ensemble_estimates.mean(axis=0)

    total_nmse = calc_NMSE(true_values, ensemble_estimates).item()
    total_rmse = calc_RMSE(true_values, ensemble_estimates).item()
    total_y2_mean = (ensemble_estimates**2).mean()

    total_uncertainties = (ensemble_estimates.var(axis=0).mean()/total_y2_mean).item()
    total_r2 = calc_R2(true_values, mean_estimates).item()
    # print(true_values.shape, true_values)

    uncertainties = calc_uncertainty(ensemble_estimates)
    rmse_pts = calc_RMSE_points(true_values, ensemble_estimates)

    # pearsonr = stats.pearsonr(np.log(rmse_pts.flatten()), np.log(uncertainties.flatten()))[0]
    try:
        pearsonr = stats.pearsonr(np.log(rmse_pts.flatten()), np.log(uncertainties.flatten()))[0]
    except:
        print(rmse_pts.flatten(), uncertainties.flatten())
        pearsonr = 0

    tmp = {}
    tmp["total_%s_nmse" %mode] = total_nmse
    tmp["total_%s_rmse" %mode] = total_rmse
    tmp["total_%s_r2" %mode] = total_r2
    tmp["total_%s_uncertain" %mode] = total_uncertainties
    
    # tmp["%s_rmse" %mode] = np.log(rmse_pts).mean()
    # tmp["%s_uncertain" %mode] = np.log(uncertainties).mean()
    tmp["%s_r" %mode] = pearsonr

    print("Total test, NMSE: %.5f  RMSE: %.5f  R2: %.5f" % (tmp["total_%s_nmse" %mode], tmp["total_%s_rmse" %mode], tmp["total_%s_r2" %mode]))
    print("Uncertainty test, Total Uncertainty: %.5f  PearsonR: %.5f" % (tmp["total_%s_uncertain" %mode], tmp["%s_r" %mode]))
    return tmp


def logging_total_test_real(ensemble_estimates, mode='drift'):
    if mode == 'diff':
        print(ensemble_estimates.mean(axis=0))

    uncertainties = calc_uncertainty(ensemble_estimates)

    tmp = {}
    tmp["%s_uncertain" %mode] = np.log(uncertainties).mean()

    print("Uncertainty test,  Uncertainty: %.5f" % (tmp["%s_uncertain" %mode]))
    return tmp


def save_checkpoint_real(state, is_best, save_path, mode='drift', model_idx = -1, use_drift = False, ihm=False):
    if use_drift and mode=='diff':
        filename = path.join(save_path, "checkpoint_diff_udT.pth.tar")
    else:
        filename = path.join(save_path, "checkpoint_%s.pth.tar" %mode)
    torch.save(state, filename)

    if is_best:
        if use_drift and mode=='diff':
            torch.save(state, path.join(save_path, "model_best_diff_udT.pth.tar"))
        
        if ihm and mode=='diff':
            torch.save(state, path.join(save_path, "model_best_diff_ihm.pth.tar"))

        else:
            torch.save(state, path.join(save_path, "model_best_%s.pth.tar" %mode))


def save_checkpoint(state, is_best, save_path, mode='drift', model_idx = -1, use_drift = False):
    if model_idx < 0:
        if use_drift and mode=='diff':
            filename = path.join(save_path, "checkpoint_diff_udT.pth.tar")
        else:
            filename = path.join(save_path, "checkpoint_%s.pth.tar" %mode)
        torch.save(state, filename)

        if is_best:
            if use_drift and mode=='diff':
                torch.save(state, path.join(save_path, "model_best_diff_udT.pth.tar"))

            else:
                torch.save(state, path.join(save_path, "model_best_%s.pth.tar" %mode))
            # shutil.copyfile(filename, path.join(save_path, "model_best_%s.pth.tar" %mode))
    else:
        filename = path.join(save_path, "checkpoint_%s_%d.pth.tar" %(mode, model_idx))
        torch.save(state, filename)
        if is_best:
            torch.save(state, path.join(save_path, "model_best_%s_%d.pth.tar" %(mode, model_idx)))