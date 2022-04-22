""" Utility functions for checkpointing.
"""

import os


def ensure_directory_exists(filename):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)


def to_cpu(ele, snapshot=None):
    """Move GPU memory snapshots to CPU
    """
    if snapshot is None:
        snapshot = {}
    if hasattr(ele, 'cpu'):
        snapshot = ele.cpu()
    elif isinstance(ele, dict):
        snapshot = {}
        for k,v in ele.items():
            snapshot[k] = None
            snapshot[k] = to_cpu(v, snapshot[k])
    elif isinstance(ele, list):
        snapshot = [None for _ in range(len(ele))]
        for idx, v in enumerate(ele):
            snapshot[idx] = to_cpu(v, snapshot[idx])

    return snapshot
