""" The module for checkpoint functions.
"""
import os
import random
from typing import Dict

import numpy as np
import torch
from periflow_sdk.checkpoint.utils import ensure_directory_exists, to_cpu


def sync_checkpoint_save(iteration: int,
                         checkpoint_name: str,
                         model=None,
                         optimizer=None,
                         lr_scheduler=None):
    # default checkpoint function assumes single model, single optimizer, single lr_scheduler
    state_dict = {}
    state_dict['iteration'] = iteration
    if model is not None:
        state_dict['model'] = model.state_dict()
    if optimzier is not None:
        state_dict['optimizer'] = optimizer.state_dict()
    if lr_scheduler is not None:
        state_dict['lr_scheduler'] = lr_scheduler.state_dict()

    ensure_directory_exists(checkpoint_name)
    torch.save(state_dict, checkpoint_name)

    # ensure persistence
    with open(checkpoint_name, 'a+') as f:
        os.fsync(f)

    # log finish file
    with open(checkpoint_name[:-18] + "_complete.log", "w") as log_f:
        log_f.write("success\n")
        os.fsync(log_f)


def sync_checkpoint_load(checkpoint_name: str,
                         model=None,
                         optimizer=None,
                         lr_scheduler=None):
    state_dict = torch.load(checkpoint_name, map_location='cpu')

    if model is not None:
        model.load_state_dict(state_dict['model'])
    if optimizer is not None:
        model.load_state_dict(state_dict['optimizer'])
    if lr_scheduler is not None:
        model.load_state_dict(state_dict['lr_scheduler'])


def save_cpu_memory(state_dict: Dict):
    """ Dump all the states inside the GPU into CPU memory.
    """
    snapshot = {}
    for name, ref in state_dict.items():
        snapshot[name] = to_cpu(ref)

    return snapshot
