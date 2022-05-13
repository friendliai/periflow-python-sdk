# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""The module for classes
"""

import math
import torch

class ResumableSequentialSampler:
    """ The resumble sequential sampler.
    """
    def __init__(self,
                 samples_per_epoch: int,
                 batch_size: int,
                 drop_last: bool = False,
                 data_parallel_rank: int = 0,
                 data_parallel_size: int = 1):
        self._samples_per_epoch = samples_per_epoch
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._consumed_samples = 0
        self._cur_index = 0

        assert batch_size % data_parallel_size == 0, "batch_size should be a multiple of data_parallel_size"
        assert samples_per_epoch % data_parallel_size == 0 or drop_last is True, \
            "ResumableSampler does not support situation where " + \
            "samples_per_epoch is not divisible by data_parallel_size and drop_last is not set " + \
            "since it may cause undesirable sample duplicates. "

        self._data_parallel_rank = data_parallel_rank
        self._data_parallel_size = data_parallel_size

    def set_processed_steps(self, processed_steps: int):
        steps_per_epoch = math.ceil(self._samples_per_epoch / self._batch_size)
        self._consumed_samples = (processed_steps // steps_per_epoch) * self._samples_per_epoch + \
            (processed_steps % steps_per_epoch) * self._batch_size
        self._cur_index = self._consumed_samples % self._samples_per_epoch

    def __len__(self):
        return self._samples_per_epoch

    def __iter__(self):
        batch = []
        for index in range(self._cur_index, self._samples_per_epoch):
            batch.append(index)
            if len(batch) == self._batch_size:
                yield batch[self._data_parallel_rank * (self._batch_size // self._data_parallel_size) \
                    :(self._data_parallel_rank + 1) * (self._batch_size // self._data_parallel_size)]
                batch = []
        if len(batch) > 0 and not self._drop_last:
            yield batch[self._data_parallel_rank * (len(batch) // self._data_parallel_size) \
                    :(self._data_parallel_rank + 1) * (len(batch) // self._data_parallel_size)]
            batch = []
        # Initialize the cur index for the future epochs
        self._cur_index = 0


class ResumableRandomSampler:
    """ The random sampler that supports deterministic & resumable random sampling.
    """
    def __init__(self,
                 samples_per_epoch: int,
                 batch_size: int,
                 drop_last: bool = False,
                 seed: int = 0,
                 data_parallel_rank: int = 0,
                 data_parallel_size: int = 1):
        self._samples_per_epoch = samples_per_epoch
        self._batch_size = batch_size
        self._cur_epoch = 0
        self._consumed_samples_cur_epoch = 0
        self._drop_last = drop_last
        self._seed = seed

        assert batch_size % data_parallel_size == 0, "batch_size should be a multiple of data_parallel_size"
        assert samples_per_epoch % data_parallel_size == 0 or drop_last is True, \
            "ResumableRandomSampler does not support situation where" + \
            "samples_per_epoch is not divisible by data_parallel_size and drop_last is not set" + \
            "since it may cause undesirable sample duplicates."

        self._data_parallel_rank = data_parallel_rank
        self._data_parallel_size = data_parallel_size

    def set_processed_steps(self, processed_steps: int):
        steps_per_epoch = math.ceil(self._samples_per_epoch / self._batch_size)
        self._cur_epoch = processed_steps // steps_per_epoch
        self._consumed_samples_cur_epoch = (processed_steps % steps_per_epoch) * self._batch_size

    def __len__(self):
        return self._samples_per_epoch

    def __iter__(self):
        g = torch.Generator()
        # We avoid seed 0 for good random generation.
        g.manual_seed(self._cur_epoch + self._seed + 1)
        random_indices = torch.randperm(self._samples_per_epoch // self._data_parallel_size, generator=g).tolist()
        start_idx = self._consumed_samples_cur_epoch // self._data_parallel_size
        offset = (self._samples_per_epoch // self._data_parallel_size) * self._data_parallel_rank
        batch = []
        for index in range(start_idx, self._samples_per_epoch // self._data_parallel_size):
            batch.append(offset + random_indices[index])
            if len(batch) == self._batch_size // self._data_parallel_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self._drop_last:
            yield batch
        # Advance one epoch for future reuses
        self._cur_epoch += 1
        self._consumed_samples_cur_epoch = 0
