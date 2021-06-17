"""The setup script."""
from setuptools import find_packages, setup

install_requires = [
    "numpy",
    "pytorch-fid",
    "torch",
    "torchvision",
]
extras_require = {
    "data": [
        "atlutils @ git+https://bbpgitlab.epfl.ch/ml/atlas-data-utils.git",
        "dvc[ssh]",
        "pillow",
        "requests",
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
