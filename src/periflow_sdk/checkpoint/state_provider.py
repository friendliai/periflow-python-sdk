""" The module for state providers.
"""


import random
from typing import Dict

import numpy as np
import torch


def default_state_provider(iteration: int,
                           modules: Dict,
                           store_random_states: bool = False) -> Dict:
    """ The default state provider function.
    If not None, it also stores optimizer and lr_scheduler states.
    If store_random_states is True, then it stores all the possible random states in order to resume dataloader states.
    """
    state_dict = {}
    state_dict['iteration'] = iteration
    for k, v in modules.items():
        assert hasattr(v, 'state_dict')
        state_dict[k] = v.state_dict()
    if store_random_states:
        state_dict['random_rng_state'] = random.getstate()
        state_dict['np_rng_state'] = np.random.get_state()
        state_dict['torch_rng_state'] = torch.get_rng_state()
        state_dict['cuda_rng_state'] = torch.cuda.get_rng_state()

    return state_dict
