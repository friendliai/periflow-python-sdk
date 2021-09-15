""" The module for checkpoint functions.
"""
import os
from typing import Dict

import torch
from .utils import ensure_directory_exists, to_cpu


def sync_checkpoint_func(state_dict: Dict, ckpt_path: str):
    """ The synchronous checkpoint function.
    """
    persist_checkpoint(state_dict, ckpt_path)


def save_cpu_memory(state_dict: Dict):
    """ Dump all the states inside the GPU into CPU memory.
    """
    snapshot = {}
    for name, ref in state_dict.items():
        snapshot[name] = to_cpu(ref)

    return snapshot


def persist_checkpoint(snapshot: Dict, ckpt_path: str):
    """ Persist checkpoints in CPU memory to disks.
    """

    ensure_directory_exists(ckpt_path)
    torch.save(snapshot, ckpt_path)

    # Ensure it's persisted
    with open(ckpt_path, 'a+') as f:
        os.fsync(f.fileno())

    # Log finish file
    with open(ckpt_path + "_complete.log", "w") as log_f:
        log_f.write("success\n")
        os.fsync(log_f.fileno())
