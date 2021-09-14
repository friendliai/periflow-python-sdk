import math
import torch

class ResumableSequentialSampler:
    def __init__(self,
                 samples_per_epoch: int,
                 processed_steps: int,
                 batch_size: int,
                 drop_last: bool):
        self._samples_per_epoch = samples_per_epoch
        self._batch_size = batch_size
        steps_per_epoch = math.ceil(samples_per_epoch / batch_size)
        self._consumed_samples = (processed_steps // steps_per_epoch) * samples_per_epoch + \
            (processed_steps % steps_per_epoch) * batch_size
        self._cur_index = self._consumed_samples % samples_per_epoch
        self._drop_last = drop_last

    def __len__(self):
        return self._samples_per_epoch

    def __iter__(self):
        batch = []
        for index in range(self._cur_index, self._samples_per_epoch):
            batch.append(index)
            if len(batch) == self._batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self._drop_last:
            yield batch
            batch = []
        # Initialize the cur index for the future epochs
        self._cur_index = 0


class ResumableRandomSampler:
    def __init__(self,
                 samples_per_epoch: int,
                 processed_steps: int,
                 batch_size: int,
                 drop_last: bool,
                 seed: int = 0):
        self._samples_per_epoch = samples_per_epoch
        self._batch_size = batch_size
        steps_per_epoch = math.ceil(samples_per_epoch / batch_size)
        self._cur_epoch = processed_steps // steps_per_epoch + 1
        self._consumed_samples = (processed_steps // steps_per_epoch) * samples_per_epoch + \
            (processed_steps % steps_per_epoch) * batch_size
        self._consumed_samples_cur_epoch = self._consumed_samples % samples_per_epoch
        self._drop_last = drop_last
        self._seed = seed

    def __len__(self):
        return self._samples_per_epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self._cur_epoch)
        random_idx = torch.randperm(self._samples_per_epoch, generator=g).tolist()
        start_idx = self._consumed_samples_cur_epoch

        batch = []
        for index in range(start_idx, self._samples_per_epoch):
            batch.append(random_idx[index])
            if len(batch) == self._batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self._drop_last:
            yield batch
        # Advance one epoch for future reuses
        self._cur_epoch += 1
        self._consumed_samples_cur_epoch = 0
