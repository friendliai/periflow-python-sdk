""" The module for state providers.
"""


import random
from typing import Dict

import numpy as np
import torch


def default_state_provider(iteration: int,
                           model,
                           optimizer = None,
                           lr_scheduler = None,
                           store_random_states: bool = False) -> Dict:
    """ The default state provider function.
    If not None, it also stores optimizer and lr_scheduler states.
    If store_random_states is True, then it stores all the possible random states in order to resume dataloader states.
    """
    state_dict = {}
    state_dict['iteration'] = iteration
    assert hasattr(model, 'state_dict')
    state_dict['model'] = model.state_dict()
    if optimizer:
        assert hasattr(optimizer, 'state_dict')
        state_dict['optimizer'] = optimizer.state_dict()
    if lr_scheduler:
        assert hasattr(lr_scheduler, 'state_dict')
        state_dict['lr_scheduler'] = lr_scheduler.state_dict()
    if store_random_states:
        state_dict['random_rng_state'] = random.getstate()
        state_dict['np_rng_state'] = np.random.get_state()
        state_dict['torch_rng_state'] = torch.get_rng_state()
        state_dict['cuda_rng_state'] = torch.cuda.get_rng_state()

    return state_dict
