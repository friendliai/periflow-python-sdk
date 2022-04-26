# Copyright (C) 2021 friendli.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
from setuptools import setup, find_packages


def read(fname):
    """
    Args:
        fname:
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def read_version():
    return read("VERSION").strip()


COMMON_DEPS = [
]

TEST_DEPS = [
    "coverage==5.5",
    "pytest==6.2.4",
    "pytest-asyncio==0.15.1",
    "pytest-cov==2.11.1",
    "pytest-benchmark==3.4.1",
    "pytest-lazy-fixture==0.6.3",
]

setup(
    name='periflow_sdk',
    version=read_version(),
    author = 'FriendliAI',
    license="Apache License 2.0",
    url = "https://github.com/friendliai/periflow-python-sdk",
    description = "PeriFlow SDK",
    packages=find_packages(include=['periflow_sdk', 'periflow_sdk.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Natural Language :: English",
    ],
    python_requires='>=3.6',
    include_package_data=True,
    install_requires=COMMON_DEPS,
    extras_require={
        "test": TEST_DEPS,
    }
)
