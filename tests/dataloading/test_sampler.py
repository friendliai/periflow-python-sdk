""" Test module for samplers
"""

import pytest
from typing import List

from periflow_sdk.dataloading.sampler import ResumableRandomSampler, ResumableSequentialSampler

@pytest.fixture
def dataset():
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_sequential_sampler_normal(dataset: List[int]):
    sampler = ResumableSequentialSampler(samples_per_epoch=len(dataset),
                                         processed_steps=0,
                                         batch_size=4,
                                         drop_last=False)
    
    # Test sequential sampling
    i = iter(sampler)
    assert next(i) == dataset[:4]
    assert next(i) == dataset[4:8]
    assert next(i) == dataset[8:10]

    # Test Stop Iteration
    with pytest.raises(StopIteration):
        next(i)

    # Test resume from start
    i = iter(sampler)
    assert next(i) == dataset[:4]

    # Test drop last
    sampler = ResumableSequentialSampler(samples_per_epoch=len(dataset),
                                         processed_steps=0,
                                         batch_size=4,
                                         drop_last=True)
    i = iter(sampler)
    assert next(i) == dataset[:4]
    assert next(i) == dataset[4:8]
    with pytest.raises(StopIteration):
        next(i)


def test_random_sampler_normal(dataset: List[int]):

    sampler = ResumableRandomSampler(samples_per_epoch=len(dataset),
                                     processed_steps=0,
                                     batch_size=4,
                                     drop_last=False)
    sampled_data = set()
    i = iter(sampler)
    sampled_data.update(next(i))
    assert len(sampled_data) == 4
    sampled_data.update(next(i))
    assert len(sampled_data) == 8
    sampled_data.update(next(i))
    assert len(sampled_data) == 10
    
    with pytest.raises(StopIteration):
        next(i)

    # Test resume
    sampled_data = set()
    i = iter(sampler)
    sampled_data.update(next(i))
    assert len(sampled_data) == 4
    sampled_data.update(next(i))
    assert len(sampled_data) == 8
    sampled_data.update(next(i))
    assert len(sampled_data) == 10

    # Test drop last
    sampler = ResumableRandomSampler(samples_per_epoch=len(dataset),
                                     processed_steps=0,
                                     batch_size=4,
                                     drop_last=True)
    sampled_data = set()
    i = iter(sampler)
    sampled_data.update(next(i))
    assert len(sampled_data) == 4
    sampled_data.update(next(i))
    assert len(sampled_data) == 8

    with pytest.raises(StopIteration):
        next(i)


def test_sequential_sampler_resume(dataset: List[int]):
    sampler = ResumableSequentialSampler(samples_per_epoch=len(dataset),
                                         processed_steps=1,
                                         batch_size=4,
                                         drop_last=False)
    
    # One batch (4 samples) is already processed
    i = iter(sampler)
    assert next(i) == dataset[4:8]
    assert next(i) == dataset[8:10]

    with pytest.raises(StopIteration):
        next(i)

    # Test resume from start
    i = iter(sampler)
    assert next(i) == dataset[:4]


def test_random_sampler_resume(dataset: List[int]):

    sampler = ResumableRandomSampler(samples_per_epoch=len(dataset),
                                     processed_steps=1,
                                     batch_size=4,
                                     drop_last=False,
                                     seed = 77)
    sampled_data = set()
    i = iter(sampler)
    sampled_data.update(next(i))
    assert len(sampled_data) == 4
    
    third_batch = next(i)

    with pytest.raises(StopIteration):
        next(i)

    sampler2 = ResumableRandomSampler(samples_per_epoch=len(dataset),
                                      processed_steps=2,
                                      batch_size=4,
                                      drop_last=False,
                                      seed = 77)
    sampled_data = set()
    i2 = iter(sampler2)
    new_third_batch = next(i2)
    sampled_data.update(new_third_batch)
    # Assures that random sampling order is just the same, after the recovery.
    assert third_batch == new_third_batch
    
    with pytest.raises(StopIteration):
        next(i2)

    # Test next epoch
    i = iter(sampler)
    i2 = iter(sampler2)

    # Note that i and i2 should generate the same batched samples
    # Also note that i and i2 should generate 10 examples in total.
    assert next(i) == next(i2)
    assert next(i) == next(i2)
    assert next(i) == next(i2)
