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
* [Data](#data)
    * [Downloading data from scratch](#downloading-data-from-scratch) 
    * [Pulling from the remote](#pulling-from-the-remote)
    * [Downloading new ISH datasets](#downloading-new-ish-datasets)
* [Examples](#examples)
    * [Pair Interpolation Models](#pair-interpolation-models)
    * [Optical Flow Models](#optical-flow-models)
    * [Predict Entire Gene Volume](#predict-entire-gene-volume)
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

## Data

The data for this project is managed using the DVC tool. There are two options to
get the data:
- Download them from scratch
- Pull the pre-downloaded data from a remote machine (on the BBP intranet)

In either case, one needs to clone the repository and install the extra `data` dependencies.
```shell
git clone https://github.com/BlueBrain/atlas-interpolation
cd atlas-interpolation/data
pip install git+https://github.com/BlueBrain/atlas-interpolation#egg=atlinter[data]
```

Pulling/Download the entire DVC is a huge amount of data (several GBs),
do not hesitate to select only the data you are interested in.

### Downloading data from scratch
Downloading data from scratch can be done easily using dvc command.
```shell
dvc repro
```
This step might take some time.

In some cases you might not need all data. Then it is possible to download unprepared
data that you need by running specific DVC stages. 
```shell
dvc repro download-nissl # To download nissl volume
dvc repro download_dataset # To download all listed genes in the dvc.yaml
dvc repro download_dataset@Vip # To download specific gene expressions datasets, in this case Vip
dvc repro download_special_dataset # To download all special datasets
```

If one wants those volumes already aligned to the Nissl Brain Atlas, one can directly 
launch one of the following commands:
```shell
dvc repro align
dvc repro align@Vip
dvc repro align_special_dataset
```
This will take some time, especially if the volumes are not already downloaded. 

### Pulling from the remote
This only works if you have access to `proj101` on BBP intranet. Otherwise, follow
the previous section [Downloading data from scratch](#downloading-data-from-scratch)
instructions.

If you are working on the BB5 please run the following commands
first:
```shell
dvc remote add --local gpfs_proj101 \
/gpfs/bbp.cscs.ch/data/project/proj101/dvc_remotes/atlas_interpolation
```

To pull all original data from the remote run
```shell
dvc pull
```

It is also possible to selectively pull data with
```shell
dvc pull <filename>.dvc
```
where `<filename>` should be replaced by one of the filenames found in the `data` directory.

### Downloading new ISH datasets
If one is interested into downloading ISH gene expressions that are not already 
part of our DVC pipeline (cf. list in [`data/dvc.yaml`](data/dvc.yaml)), one can
directly add the name of the gene of interest in the `foreach` list and launch:
```shell
dvc repro download_dataset@NEW_GENE
dvc repro align@NEW_GENE
```

## Examples

Here are the different experiment one can do with `atlinter` package:
- One can predict one/several images between a pair of images thanks to pair interpolation models
- One can predict optical flow between any pair of images and use it to create a new image
- One can predict a given slice or an entire gene volume.

Note that every models accept RGB images of shape `(height, width, 3)`
and grayscale images of shape `(height, width)`.

### Pair Interpolation Models

#### Setup

To use one of the pair interpolation models integrated in `atlinter` package,
one needs first to download/pull the specific checkpoints of the model
```shell
cd data

dvc pull checkpoints/rife.dvc # RIFE model
dvc pull checkpoints/cain.dvc # CAIN model
```

#### Example Code

To make it run correctly, please be at the root folder of the project.

```python
import numpy as np

from atlinter.vendor.rife.RIFE_HD import Model as RifeModel
from atlinter.vendor.rife.RIFE_HD import device as rife_device
from atlinter.pair_interpolation import PairInterpolate, RIFEPairInterpolationModel

# Instantiate the Pair Interpolation model (in this case: RIFE)
rife_model = RifeModel()
rife_model.load_model("data/checkpoints/rife/", -1)
rife_model.eval()
rife_interpolation_model = RIFEPairInterpolationModel(rife_model, rife_device)

# Predict middle image between img1 and img2
img1 = np.random.rand(100, 200, 3) # replace by real section image
img2 = np.random.rand(100, 200, 3) # replace by real section image
img1, img2 = rife_interpolation_model.before_interpolation(img1=img1, img2=img2)
img_middle = rife_interpolation_model.interpolate(img1=img1, img2=img2)
img_middle = rife_interpolation_model.after_interpolation(img_middle)

# If you want to predict several images between img1 and img2
img1 = np.random.rand(100, 200, 3) # replace by real section image
img2 = np.random.rand(100, 200, 3) # replace by real section image
interpolated_imgs = PairInterpolate(n_repeat=3)(img1, img2, rife_interpolation_model)
``` 

### Optical Flow Models

#### Setup

If you want to use one of the optical flow models, please make sure that `optical`
extra dependencies are installed.
```shell
pip install git+https://github.com/BlueBrain/atlas-interpolation#egg=atlinter[optical]
```

One also needs to download/pull the specific checkpoints of the model:
```shell
cd data

dvc pull checkpoints/RAFT.dvc               # RAFT model
dvc pull checkpoints/maskflownet.params.dvc # MaskFlowNet model
```

#### Example Code

```python
from atlinter.optical_flow import MaskFlowNet

# Instantiate the Optical Flow model (in this case: MaskFlowNet)
checkpoint_path = "data/checkpoints/maskflownet.params"
net = MaskFlowNet(checkpoint_path)

# Predict flow between img1 and img2
img1 = np.random.rand(100, 200, 3) # replace by real section image
img2 = np.random.rand(100, 200, 3) # replace by real section image
img3 = np.random.rand(100, 200, 3) # replace by real section image
img1, img2 = net.preprocess_images(img1=img1, img2=img2)
predicted_flow = net.predict_flow(img1=img1, img2=img2)

# If you want to predict images thanks to optical flow
predicted_img = net.warp_image(predicted_flow, img3)
``` 

### Predict entire gene volume

#### Setup

Please make sure to have the dataset `Vip` locally before running the code snippet.
If it is not the case, please download it by following the [Data](#data) section instructions.

#### Example Code

```python
import json

import numpy as np

from atlinter.data import GeneDataset
from atlinter.pair_interpolation import (
   GeneInterpolate,
   RIFEPairInterpolationModel,
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
