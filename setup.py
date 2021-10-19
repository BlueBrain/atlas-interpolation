# Copyright 2021, Blue Brain Project, EPFL
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
"""The setup script."""
from setuptools import find_packages, setup

install_requires = [
    "mxnet",
    "numpy",
    "pytorch-fid",
    "pyyaml",
    "scipy",
    "torch",
    "torchvision",
]
extras_require = {
    "data": [
        "atlannot @ git+https://bbpgitlab.epfl.ch/project/proj101/atlas_annotation.git",
        "atldld==0.2.2",
        "dvc[ssh]",
        "pillow",
        "pynrrd",
        "requests",
        "scikit-image",
    ],
    "dev": [
        "bandit",
        "black",
        "flake8",
        "flake8-bugbear",
        "flake8-comprehensions",
        "flake8-docstrings",
        "isort",
        "pytest",
        "pytest-cov",
        "tox",
    ],
    "optical": [
        "opencv-python",
        "flow_vis",
        "moviepy",
    ],
}

setup(
    name="atlinter",
    use_scm_version={
        "write_to": "src/atlinter/version.py",
        "write_to_template": '"""The package version."""\n__version__ = "{version}"\n',
        "local_scheme": "no-local-version",
    },
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires="~=3.7.0",
    install_requires=install_requires,
    extras_require=extras_require,
)
