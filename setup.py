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

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

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
        "atlannot",
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
        "py-vendor==0.1.2",
        "pytest",
        "pytest-cov",
        "tox",
    ],
    "docs": [
        "sphinx",
        "sphinx-bluebrain-theme",
    ],
    "optical": [
        "opencv-python",
        "flow_vis",
        "moviepy",
    ],
}

setup(
    name="atlinter",
    author="Blue Brain Project, EPFL",
    license="Apache-2.0",
    use_scm_version={
        "write_to": "src/atlinter/version.py",
        "write_to_template": '"""The package version."""\n__version__ = "{version}"\n',
        "local_scheme": "no-local-version",
    },
    description="Interpolate missing section images in gene expression volumes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BlueBrain/atlas-interpolation",
    project_urls={
        "Source": "https://github.com/BlueBrain/atlas-interpolation",
        "Tracker": "https://github.com/BlueBrain/atlas-interpolation/issues",
    },
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.7",
    ],
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires="~=3.7.0",
    install_requires=install_requires,
    extras_require=extras_require,
)
