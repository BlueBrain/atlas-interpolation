# Atlas Interpolation

* [Installation](#installation)
    * [Installation from source](#installation-from-source)
    * [Installation for development](#installation-for-development)
* [Examples](#examples)
* [Vendors](#vendors)
* [Funding & Acknowledgment](#funding--acknowledgment)

## Installation

### Installation from source
If you want to try the latest version, you can install from source.

```shell
pip install git+https://github.com/BlueBrain/atlas-interpolation
```

### Installation for development
If you want a dev install, you should install the latest version from 
source with all the extra requirements for running test.

```shell
git clone https://github.com/BlueBrain/atlas-interpolation
cd atlas-interpolation
pip install -e '.[data, dev, optical]'
```

## Examples

## Vendors
Some dependencies are not available as packages and therefore had to be vendored.

See the `get_vendors.sh` script for the details on vendored data.

## Funding & Acknowledgment

The development of this software was supported by funding to the Blue Brain Project,
a research center of the École polytechnique fédérale de Lausanne (EPFL),
from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) 2021 Blue Brain Project/EPFL
