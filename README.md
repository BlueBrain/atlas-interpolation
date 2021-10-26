# Atlas Interpolation

The Allen Brain Institute hosts a rich database of mouse brain imagery. It
contains a large number of gene expression datasets obtained
through the in situ hybridization (ISH) staining. While for a given gene
a number of datasets corresponding to different specimen can be found, each of
these datasets only contains sparse section images that do not form a
continuous volume. This package explores techniques that allow to interpolate
the missing slices and thus reconstruct whole gene expression volumes.

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

Here are the different experiment one can do with `atlinter` package:

- One can predict one/several images between a pair of images thanks to pair interpolation models:

```python
from atlinter.vendor.rife.RIFE_HD import Model as RifeModel
from atlinter.vendor.rife.RIFE_HD import device as rife_device
from atlinter.pair_interpolation import PairInterpolate, RIFEPairInterpolationModel

# Instantiate the Pair Interpolation model (in this case: RIFE)
rife_model = RifeModel()
rife_model.load_model("/path/to/train_log", -1)
rife_model.eval()
rife_interpolation_model = RIFEPairInterpolationModel(rife_model, rife_device)

# Predict middle image between img1 and img2
img1 = ...
img2 = ...
img1, img2 = rife_interpolation_model.before_interpolation(img1=img1, img2=img2)
img_middle = rife_interpolation_model.interpolate(img1=img1, img2=img2)
img_middle =rife_interpolation_model.after_interpolation(img_middle)

# If you want to predict several images between img1 and img2
interpolated_imgs = PairInterpolate(n_repeat=3)(img1, img2, rife_interpolation_model)
``` 

- One can predict optical flow between any pair of images and use it to create. Please make sure
that `optical` extra dependencies are installed.
```shell
pip install git+https://github.com/BlueBrain/atlas-interpolation#egg=atlinter[optical]
```

```python
from atlinter.optical_flow import MaskFlowNet

# Instantiate the Optical Flow model (in this case: MaskFlowNet)
checkpoint_path = "data/checkpoints/maskflownet.params"
net = MaskFlowNet(checkpoint_path)

# Predict flow between img1 and img2
img1 = ...
img2 = ...
img3 = ...
img1, img2 = net.preprocess_images(img1=img1, img2=img2)
predicted_flow = net.predict_flow(img1=img1, img2=img2)

# If you want to predict images thanks to optical flow
predicted_img = net.warp_image(predicted_flow, img3)
``` 

- One can predict a given slice or an entire gene volume:
```python
import json

import numpy as np

from atlinter.data import GeneDataset
from atlinter.pair_interpolation import (
GeneInterpolate, 
RIFEPairInterpolationModel
)
from atlinter.vendor.rife.RIFE_HD import Model as RifeModel
from atlinter.vendor.rife.RIFE_HD import device as rife_device

# 1.  Prepare dataset
# 1.a Load gene and section numbers
gene = np.load("data/sagittal/Vip/1102.npy")
with open("data/sagittal/Vip/1102.json") as f:
    metadata = json.load(f)
section_numbers = [int(s) for s in metadata["section_numbers"]]

# 1.b Instantiate GeneDataset
gene_dataset = GeneDataset(
  gene,
  section_numbers,
  volume_shape=(528, 320, 456, 3),
  axis="sagittal"
)

# 2. Choose and instantiate the model (for example RIFE)
rife_model = RifeModel()
rife_model.load_model("data/checkpoints/rife/", -1)
rife_model.eval()
rife_interpolation_model = RIFEPairInterpolationModel(rife_model, rife_device)

# 3. Instantiate GeneInterpolate and predict specific slice or entire volume
gene_interpolate = GeneInterpolate(gene_dataset, rife_interpolation_model)
predicted_slice = gene_interpolate.predict_slice(10)
predicted_volume = gene_interpolate.predict_volume()
```

## Data

The data for this project is managed using the DVC tool.

All data is stored in the `data` directory. DVC is similar to git. To pull all original
data from the remote run
```shell
cd data
dvc pull
```

It is also possible to selectively pull data with
```shell
cd data
dvc pull <filename>.dvc
```
where `<filename>` should be replaced by one of the filenames found in the `data` directory.
See the `data/README.md` file for the description of different data files.

## Vendors
Some dependencies are not available as packages and therefore had to be
vendored. The vendoring is done using the
[`py-vendor`](https://pypi.org/project/py-vendor/) utility. It's installed
automatically together with the `dev` extras. You can also install it by hand
via `pip install py-vendor==0.1.2`.

The vendoring is then done using the following command (add `--force` to
overwrite existing folders):
```shell
py-vendor run --config py-vendor.yaml
```
See the `py-vendor.yaml` file for details on the vendor sources and files.

## Funding & Acknowledgment

The development of this software was supported by funding to the Blue Brain Project,
a research center of the École polytechnique fédérale de Lausanne (EPFL),
from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) 2021 Blue Brain Project/EPFL
