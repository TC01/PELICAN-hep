#!/usr/bin/env python

import argparse
import os
import logging
import random
import sys

# Numpy
import numpy
import scipy

# PyTorch imports.
import pytorch_nndct
import torch
from torch.nn import functional
from torch.utils.data import DataLoader
from torch import optim
from pytorch_nndct.apis import Inspector
from pytorch_nndct.apis import torch_quantizer

# is this needed?
from torchvision import transforms

# PELICAN internals.
from src.models import PELICANClassifier, PELICANMultiClassifier
from src.models import tests
from src.trainer import Trainer
from src.trainer import init_argparse, init_file_paths, init_logger, init_cuda, logging_printout, fix_args
from src.trainer import init_optimizer, init_scheduler
from src.models.metrics_classifier import metrics, minibatch_metrics, minibatch_metrics_string

# Python 3.9 backwards compatibility...
from src import argparse9 as argparse

from src.dataloaders import initialize_datasets, collate_fn

sys.setrecursionlimit(10000)

# Ugh, logging
logger = logging.getLogger('')

# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000, sci_mode=False)

# Figure out how to hack this in later.
quant_dir ='./build/quantized'
quant_mode = 'calib'

torchfile = '515_100p-b_best.pt'

def main():

    # Initialize arguments
    # Except this is a bad idea because this script will have different things.
    args = init_argparse()

    # Initialize file paths
    args = init_file_paths(args)

    # Fix possible inconsistencies in arguments
    args = fix_args(args)

    # Initialize logger
    init_logger(args)

    # Write input paramaters and paths to log
    logging_printout(args)

    # Initialize dataloder
    #if args.fix_data:
    #    torch.manual_seed(165937750084982)
    args, datasets = initialize_datasets(args, args.datadir, num_pts=None)

    # Fix possible inconsistencies in arguments
    args = fix_args(args)

    # Initialize device and data type
    device, dtype = init_cuda(args)
    dtype = torch.float32

    # Construct PyTorch dataloaders from datasets
    collate = lambda data: collate_fn(data, scale=args.scale, nobj=args.nobj, add_beams=args.add_beams, beam_mass=args.beam_mass, read_pid=args.read_pid)
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.num_workers,
                                     worker_init_fn=seed_worker,
                                     collate_fn=collate)
                   for split, dataset in datasets.items()}

    # Torch.load the file that needs to be loaded.
    model_state = torch.load(torchfile, map_location=torch.device('cpu'))

    # Initialize model
    #model = PELICANMultiClassifier(args.num_channels_scalar, args.num_channels_m, args.num_channels_2to2, args.num_channels_out, args.num_channels_m_out, num_classes=5,
    #                  activate_agg=args.activate_agg, activate_lin=args.activate_lin,
    #                  activation=args.activation, add_beams=args.add_beams, read_pid=args.read_pid, config=args.config, config_out=args.config_out, average_nobj=args.nobj_avg,
    #                  factorize=args.factorize, masked=args.masked,
    #                  activate_agg_out=args.activate_agg_out, activate_lin_out=args.activate_lin_out, mlp_out=args.mlp_out,
    #                  scale=args.scale, irc_safe=args.irc_safe, dropout = args.dropout, drop_rate=args.drop_rate, drop_rate_out=args.drop_rate_out, batchnorm=args.batchnorm,
    #                  device=device, dtype=dtype)
    model = PELICANClassifier(args.num_channels_scalar, [[]], [1], args.num_channels_out, [15],
                      activate_agg=args.activate_agg, activate_lin=args.activate_lin,
                      activation=args.activation, add_beams=args.add_beams, read_pid=args.read_pid, config=args.config, config_out=args.config_out, average_nobj=args.nobj_avg,
                      factorize=args.factorize, masked=args.masked,
                      activate_agg_out=args.activate_agg_out, activate_lin_out=args.activate_lin_out, mlp_out=args.mlp_out,
                      scale=args.scale, irc_safe=args.irc_safe, dropout = args.dropout, drop_rate=args.drop_rate, drop_rate_out=args.drop_rate_out, batchnorm=args.batchnorm,
                      device=device, dtype=dtype)

#    print(model_state['args'])

    model.load_state_dict(model_state['model_state'])

    model.to(device)

    # Initialize the scheduler and optimizer
    #if args.task.startswith('eval'):
    #    optimizer = scheduler = None
    #    restart_epochs = []
    #    summarize = False
    #else:
    #    optimizer = init_optimizer(args, model, len(dataloaders['train']))
    #    scheduler, restart_epochs = init_scheduler(args, optimizer)

    # Apply the covariance and permutation invariance tests.
#    if args.test:
#        tests(model, dataloaders['train'], args, tests=['gpu','irc', 'permutation'])

    # we need to generate an input tensor in the right shape.
    # Rather than do that just pull an event off the dataloader. probably this is horrible for ML reasons
    input_shape = next(iter(dataloaders['train']))
#    del input_shape['Nobj']
    if 'is_signal' in input_shape.keys():
        del input_shape['is_signal']

    print(input_shape.keys())
    if 'scalars' in input_shape.keys():
        input_shape_list = [input_shape['Pmu'], input_shape['particle_mask'], input_shape['edge_mask'], input_shape['scalars']]
    else:
        input_shape_list = [input_shape['Pmu'], input_shape['particle_mask'], input_shape['edge_mask']]

    # Inspect the model.
    inspector = Inspector("DPUCVDX8G_ISA3_C32B6")
    inspector.inspect(model, (input_shape), device=device, image_format='png')

    # nndct_macs, nndct_params = summary.model_complexity(model, input, return_flops=False, readable=False, print_model_analysis=True)
    quantizer = torch_quantizer(quant_mode, model, (input_shape), output_dir = quant_dir, device=device)
    model = quantizer.quant_model

    # Hopefully this, like, is actually equivalent?
    # Testing is overrated.
    print("Running tests")
    tests(model, dataloaders['train'], args, tests=['gpu','irc', 'permutation'])

    # "calib" step.
    quantizer.export_quant_config()

    # "deploy" step. The Xilinx scripts don't do all these things at once.
    quantizer.export_xmodel(quant_dir, deploy_check=True)
    quantizer.export_torch_script(output_dir=quant_dir)
    quantizer.export_onnx_model(output_dir=quant_dir)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def test(model, device, test_loader, deploy=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += functional.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if deploy:
                return

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    main()
