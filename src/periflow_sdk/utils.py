"""Helper functions
"""
import os
from dataclasses import dataclass


_AXIS_TO_DEGREE = {
    x: f"{x.upper()}_DEGREE" for x in ["pp", "dp", "mp"]
}


_AXIS_TO_RANK_ATTR = {
    x: f"{x}_rank" for x in ["pp", "dp", "mp"]
}


@dataclass
class DistributeConfig:
    local_rank: int
    rank: int

    def __post_init__(self):
        self.pp_rank: int = 1
        self.mp_rank: int = 1
        self.dp_rank: int = 1


def ensure_divisibility(numerator: int, denominator: int):
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"


def _get_parallel_degree(parallel_axis: str) -> int:
    degree_key = _AXIS_TO_DEGREE[parallel_axis]
    return int(os.environ[degree_key])


def _get_parallel_rank_attribute(parallel_axis: str) -> str:
    return _AXIS_TO_RANK_ATTR[parallel_axis]


def ensure_valid_parallelism_config(dist_config: DistributeConfig):
    """Sanity check for distributed parallelism configuration

    Assumptions
        - PP_DEGREE x DP_DEGREE x MP_DEGREE == WORLD_SIZE always
        - WORLD_SIZE % $(ANY_PARALLEL_DEGREE) == 0
    """
    dp_degree = os.environ.get("DP_DEGREE", None)
    mp_degree = os.environ.get("MP_DEGREE", None)
    pp_degree = os.environ.get("PP_DEGREE", None)
    parallelism_order = os.environ.get("PARALLELISM_ORDER", None)

    if all(x is None for x in [dp_degree, mp_degree, pp_degree, parallelism_order]):
        # TODO (taebum): should work when dist config is not set
        raise NotImplementedError

    if any(x is None for x in [dp_degree, mp_degree, pp_degree, parallelism_order]):
        none_keys = [x for x in ["DP_DEGREE", "MP_DEGREE", "PP_DEGREE", "PARALLELISM_ORDER"] \
                     if os.environ.get(x, None) is None]
        raise RuntimeError(f"Following parallelism elements are required: {none_keys}")

    dp_degree = int(dp_degree)
    mp_degree = int(mp_degree)
    pp_degree = int(pp_degree)

    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == dp_degree * mp_degree * pp_degree

    parallelism_order = parallelism_order.split(",")
    assert set(parallelism_order) == {"pp", "dp", "mp"}, "Parallelism order should have `pp / dp / mp`"

    prev_strides = strides = world_size
    for parallel_axis in parallelism_order:
        strides = strides // _get_parallel_degree(parallel_axis)
        for pivot_rank in range(0, world_size, prev_strides):
            for start_rank in range(pivot_rank, pivot_rank + strides):
                for parallel_axis_rank, rank in enumerate(range(start_rank, start_rank + prev_strides, strides)):
                    if rank == dist_config.rank:
                        # current process' rank
                        attr = _get_parallel_rank_attribute(parallel_axis)
                        setattr(dist_config, attr, parallel_axis_rank)

        prev_strides = strides
