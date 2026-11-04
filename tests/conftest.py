"""Shared fixtures: compiled kernels for every example, built once per session.

The kernels are built into a fresh temporary cache, so the tests always
exercise code generation and compilation from scratch.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from auto_optimal_control import examples
from auto_optimal_control.build import build_kernels
from auto_optimal_control.numeric import NumericSystem
from auto_optimal_control.pontryagin import derive


# Examples whose singular arcs are derived (exact singular-arc shooting).
SINGULAR = {"singular_regulator", "goddard"}


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="run tests marked slow (the literature benchmarks)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def kernel_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("aoc-cache")


@pytest.fixture(scope="session")
def compiled(kernel_cache):
    """name -> (PontryaginSystem, NumericSystem, Kernels)."""
    systems = {name: derive(problem, singular=name in SINGULAR)
               for name, problem in examples.all_examples().items()}
    # Each build is a compiler subprocess, so threads build in parallel.
    with ThreadPoolExecutor(len(systems)) as pool:
        kernels = {name: pool.submit(build_kernels, pmp, name, cache_dir=kernel_cache)
                   for name, pmp in systems.items()}
        return {name: (pmp, NumericSystem(pmp), kernels[name].result())
                for name, pmp in systems.items()}
