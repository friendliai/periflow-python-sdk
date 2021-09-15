""" Test module for samplers
"""

import pytest

from periflow_sdk.dataloading.sampler import ResumableRandomSampler, ResumableSequentialSampler

@pytest.fixture
def dataset():
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_sequential_sampler_distributed(dataset):

    sampler = ResumableSequentialSampler(samples_per_epoch=len(dataset),
                                         processed_steps=0,
                                         batch_size=4,
                                         drop_last=False,
                                         data_parallel_rank = 0,
                                         data_parallel_size = 2)

    sampler2 = ResumableSequentialSampler(samples_per_epoch=len(dataset),
                                          processed_steps=0,
                                          batch_size=4,
                                          drop_last=False,
                                          data_parallel_rank = 1,
                                          data_parallel_size = 2)

    i = iter(sampler)
    i2 = iter(sampler2)
    assert next(i) == dataset[0:2]
    assert next(i2) == dataset[2:4]
    assert next(i) == dataset[4:6]
    assert next(i2) == dataset[6:8]
    assert next(i) == dataset[8:9]
    assert next(i2) == dataset[9:10]

    with pytest.raises(StopIteration):
        next(i)

    with pytest.raises(StopIteration):
        next(i2)

    # Resume a sampler
    sampler3 = ResumableSequentialSampler(samples_per_epoch=len(dataset),
                                          processed_steps=1,
                                          batch_size=4,
                                          drop_last=False,
                                          data_parallel_rank = 0,
                                          data_parallel_size = 2)
    i3 = iter(sampler3)
    assert next(i3) == dataset[4:6]
    assert next(i3) == dataset[8:9]

    # Resume a sampler after some epochs
    sampler4 = ResumableSequentialSampler(samples_per_epoch=len(dataset),
                                          processed_steps=7,
                                          batch_size=4,
                                          drop_last=False,
                                          data_parallel_rank = 0,
                                          data_parallel_size = 2)

    i4 = iter(sampler4)
    assert next(i4) == dataset[4:6]
    assert next(i4) == dataset[8:9]



def test_random_sampler_distributed(dataset):

    sampler = ResumableRandomSampler(samples_per_epoch=len(dataset),
                                     processed_steps=0,
                                     batch_size=4,
                                     drop_last=False,
                                     seed = 77,
                                     data_parallel_rank = 0,
                                     data_parallel_size = 2)

    sampler2 = ResumableRandomSampler(samples_per_epoch=len(dataset),
                                     processed_steps=0,
                                     batch_size=4,
                                     drop_last=False,
                                     seed = 77,
                                     data_parallel_rank = 1,
                                     data_parallel_size = 2)

    sampled_data = set()
    i = iter(sampler)
    i2 = iter(sampler2)
    sampled_data.update(next(i))
    assert len(sampled_data) == 2
    sampled_data.update(next(i2))
    assert len(sampled_data) == 4

    second_local_batch = next(i)
    sampled_data.update(second_local_batch)
    assert len(sampled_data) == 6
    sampled_data.update(next(i2))
    assert len(sampled_data) == 8

    third_local_batch = next(i)
    sampled_data.update(third_local_batch)
    assert len(sampled_data) == 9
    sampled_data.update(next(i2))
    assert len(sampled_data) == 10

    with pytest.raises(StopIteration):
        next(i)

    with pytest.raises(StopIteration):
        next(i2)

    # Resume a sampler
    sampler3 = ResumableRandomSampler(samples_per_epoch=len(dataset),
                                      processed_steps=1,
                                      batch_size=4,
                                      drop_last=False,
                                      seed = 77,
                                      data_parallel_rank = 0,
                                      data_parallel_size = 2)
    i3 = iter(sampler3)
    resumed_second_local_batch = next(i3)
    resumed_third_local_batch = next(i3)

    assert second_local_batch == resumed_second_local_batch
    assert third_local_batch == resumed_third_local_batch
