import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader


def data_loader(train_data, valid_data, test_data, batch_size, test_batch_size, mode='drift'):
    # (# of trajectories, length of trajectory, # of dimension)
    data_dim = train_data.shape[-1]

    if mode == 'drift':
        train_data = TensorDataset(torch.FloatTensor(train_data[:, :-1].reshape(-1, data_dim)), 
                                torch.FloatTensor(np.diff(train_data, axis=-2).reshape(-1, data_dim)))
        valid_data = TensorDataset(torch.FloatTensor(valid_data[:, :-1].reshape(-1, data_dim)), 
                                torch.FloatTensor(np.diff(valid_data, axis=-2).reshape(-1, data_dim)))
        test_data = TensorDataset(torch.FloatTensor(test_data[:, :-1].reshape(-1, data_dim)), 
                                torch.FloatTensor(np.diff(test_data, axis=-2).reshape(-1, data_dim)))

    elif mode == 'diff':
        tril_ind = torch.tril_indices(data_dim, data_dim)

        dx2 = torch.FloatTensor(np.diff(train_data, axis=-2).reshape(-1, data_dim))
        dx2_valid = torch.FloatTensor(np.diff(valid_data, axis=-2).reshape(-1, data_dim))
        dx2_test = torch.FloatTensor(np.diff(test_data, axis=-2).reshape(-1, data_dim))

        dx2 = torch.einsum('bi,bj->bij', dx2, dx2)
        dx2_valid = torch.einsum('bi,bj->bij', dx2_valid, dx2_valid)
        dx2_test = torch.einsum('bi,bj->bij', dx2_test, dx2_test)
        
        train_data = TensorDataset(torch.FloatTensor(train_data[:, :-1].reshape(-1, data_dim)), 
                                torch.FloatTensor(dx2[:, tril_ind[0], tril_ind[1]]))
        valid_data = TensorDataset(torch.FloatTensor(valid_data[:, :-1].reshape(-1, data_dim)), 
                                torch.FloatTensor(dx2_valid[:, tril_ind[0], tril_ind[1]]))
        test_data = TensorDataset(torch.FloatTensor(test_data[:, :-1].reshape(-1, data_dim)), 
                                torch.FloatTensor(dx2_test[:, tril_ind[0], tril_ind[1]]))

    train_loader = DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
            )
    valid_loader = DataLoader(
                valid_data,
                batch_size=test_batch_size,
                shuffle=False,
                pin_memory=True,
            )
    test_loader = DataLoader(
                test_data,
                batch_size=test_batch_size,
                shuffle=False,
                pin_memory=True,
            )
        
    return train_loader, valid_loader, test_loader


def underdamped_data_loader(train_data, valid_data, test_data, batch_size, test_batch_size, mode='drift'):
    # (# of trajectories, length of trajectory, # of dimension)
    data_dim = train_data.shape[-1]//2

    train_x = train_data[:, :, :data_dim]
    valid_x = valid_data[:, :, :data_dim]
    test_x = test_data[:, :, :data_dim]

    train_v = train_data[:, :, data_dim:]
    valid_v = valid_data[:, :, data_dim:]
    test_v = test_data[:, :, data_dim:]

    delta_v = train_v[:, 1:] - train_v[:, :-1]
    valid_delta_v = valid_v[:, 1:] - valid_v[:, :-1]
    test_delta_v = test_v[:, 1:] - test_v[:, :-1]

    # delta_v = (train_data[:, 2:] - 2 * train_data[:, 1:-1] + train_data[:, :-2])/dt
    # valid_delta_v = (valid_data[:, 2:] - 2 * valid_data[:, 1:-1] + valid_data[:, :-2])/dt
    # test_delta_v = (test_data[:, 2:] - 2 * test_data[:, 1:-1] + test_data[:, :-2])/dt 

    if mode == 'drift':
        train_dataset = TensorDataset(torch.FloatTensor(torch.cat((train_x[:, :-1], train_v[:, :-1]), axis=2).reshape(-1, 2*data_dim)), 
                                torch.FloatTensor(delta_v.reshape(-1, data_dim)))

        valid_dataset = TensorDataset(torch.FloatTensor(torch.cat((valid_x[:, :-1], valid_v[:, :-1]), axis=2).reshape(-1, 2*data_dim)), 
                                torch.FloatTensor(valid_delta_v.reshape(-1, data_dim)))

        test_dataset = TensorDataset(torch.FloatTensor(torch.cat((test_x[:, :-1], test_v[:, :-1]), axis=2).reshape(-1, 2*data_dim)), 
                                torch.FloatTensor(test_delta_v.reshape(-1, data_dim)))

    elif mode == 'diff':
        tril_ind = torch.tril_indices(data_dim, data_dim)

        dv2 = delta_v.reshape(-1, data_dim)
        dv2 = torch.einsum('bi,bj->bij', dv2, dv2)
        train_dataset = TensorDataset(torch.FloatTensor(torch.cat((train_x[:, :-1], train_v[:, :-1]), axis=2).reshape(-1, 2*data_dim)), 
                                torch.FloatTensor(dv2[:, tril_ind[0], tril_ind[1]]))        

        dv2_valid = valid_delta_v.reshape(-1, data_dim)
        dv2_valid = torch.einsum('bi,bj->bij', dv2_valid, dv2_valid)               
        valid_dataset = TensorDataset(torch.FloatTensor(torch.cat((valid_x[:, :-1], valid_v[:, :-1]), axis=2).reshape(-1, 2*data_dim)), 
                                torch.FloatTensor(dv2_valid[:, tril_ind[0], tril_ind[1]]))

        dv2_test = test_delta_v.reshape(-1, data_dim)
        dv2_test = torch.einsum('bi,bj->bij', dv2_test, dv2_test)               
        test_dataset = TensorDataset(torch.FloatTensor(torch.cat((test_x[:, :-1], test_v[:, :-1]), axis=2).reshape(-1, 2*data_dim)), 
                                torch.FloatTensor(dv2_test[:, tril_ind[0], tril_ind[1]]))

    train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
            )
    valid_loader = DataLoader(
                valid_dataset,
                batch_size=test_batch_size,
                shuffle=False,
                pin_memory=True,
            )
    test_loader = DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                shuffle=False,
                pin_memory=True,
            )
        
    return train_loader, valid_loader, test_loader


def underdamped_time_data_loader(train_data, valid_data, test_data, batch_size, test_batch_size, mode='drift'):
    data_dim = (train_data.shape[-1] - 1)//2

    # divide datasets (train, valid, and test) into two sets, x and v
    train_x = train_data[:, :, :data_dim]
    valid_x = valid_data[:, :, :data_dim]
    test_x = test_data[:, :, :data_dim]

    train_v = train_data[:, :, data_dim:-1]
    valid_v = valid_data[:, :, data_dim:-1]
    test_v = test_data[:, :, data_dim:-1]

    delta_v = train_v[:, 1:] - train_v[:, :-1]
    valid_delta_v = valid_v[:, 1:] - valid_v[:, :-1]
    test_delta_v = test_v[:, 1:] - test_v[:, :-1]

    # make datasets which will be fed to the LNN
    train_dataset_states = torch.FloatTensor(train_data[:, :-1].reshape(-1, 2 * data_dim + 1))
    valid_dataset_states = torch.FloatTensor(valid_data[:, :-1].reshape(-1, 2 * data_dim + 1))
    test_dataset_states = torch.FloatTensor(test_data[:, :-1].reshape(-1, 2 * data_dim + 1))

    # For the drift mode, labels are \Delta v
    if mode == 'drift':
        train_dataset = TensorDataset(train_dataset_states, 
                                torch.FloatTensor(delta_v.reshape(-1, data_dim)))

        valid_dataset = TensorDataset(valid_dataset_states, 
                                torch.FloatTensor(valid_delta_v.reshape(-1, data_dim)))

        test_dataset = TensorDataset(test_dataset_states, 
                                torch.FloatTensor(test_delta_v.reshape(-1, data_dim)))

    # For the diff mode, labels are \Delta v \Delta v^T (= dv2)
    elif mode == 'diff':
        tril_ind = torch.tril_indices(data_dim, data_dim)

        dv2 = delta_v.reshape(-1, data_dim)
        dv2 = torch.einsum('bi,bj->bij', dv2, dv2)
        train_dataset = TensorDataset(train_dataset_states, 
                                torch.FloatTensor(dv2[:, tril_ind[0], tril_ind[1]]))        

        dv2_valid = valid_delta_v.reshape(-1, data_dim)
        dv2_valid = torch.einsum('bi,bj->bij', dv2_valid, dv2_valid)
        valid_dataset = TensorDataset(valid_dataset_states, 
                                torch.FloatTensor(dv2_valid[:, tril_ind[0], tril_ind[1]]))

        dv2_test = test_delta_v.reshape(-1, data_dim)
        dv2_test = torch.einsum('bi,bj->bij', dv2_test, dv2_test)
        test_dataset = TensorDataset(test_dataset_states, 
                                torch.FloatTensor(dv2_test[:, tril_ind[0], tril_ind[1]]))

    # make DataLoaders
    train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
            )
    valid_loader = DataLoader(
                valid_dataset,
                batch_size=test_batch_size,
                shuffle=False,
                pin_memory=True,
            )
    test_loader = DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                shuffle=False,
                pin_memory=True,
            )
        
    return train_loader, valid_loader, test_loader
