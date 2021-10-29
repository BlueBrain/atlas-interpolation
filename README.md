# Atlas Interpolation

The Allen Brain Institute hosts a rich database of mouse brain imagery. It
contains a large number of gene expression datasets obtained
through the in situ hybridization (ISH) staining. While for a given gene
a number of datasets corresponding to different specimen can be found, each of
these datasets only contains sparse section images that do not form a
continuous volume. This package explores techniques that allow to interpolate
the missing slices and thus reconstruct whole gene expression volumes.

* [Installation](#installation)
    * [Python Version and Environment](#python-version-and-environment)
    * [Install "Atlas Interpolation"](#install-atlas-interpolation)
* [Data](#data)
    * [Remote Storage Access](#remote-storage-access) 
    * [Model Checkpoints](#model-checkpoints)
    * [Section Images and Datasets](#section-images-and-datasets)
    * [New ISH datasets (advanced, optional)](#new-ish-datasets-advanced-optional)
* [Examples](#examples)
    * [Pair Interpolation Models](#pair-interpolation-models)
    * [Optical Flow Models](#optical-flow-models)
    * [Predict Entire Gene Volume](#predict-entire-gene-volume)
* [Vendors](#vendors)
* [Funding & Acknowledgment](#funding--acknowledgment)

## Installation
### Python Version and Environment
Note that due to some of our dependencies we're currently limited to python
version `3.7`. Please make sure you set up a virtual environment with that
version before trying to install this library. If you're unsure how to do that
please have a look at [conda](https://docs.conda.io) or
[pyenv](https://github.com/pyenv/pyenv).

If you are part of the Blue Brain Project and are working on the BB5 you can
find the correct python version in the archive modules between `archive/2020-02`
and `archive/2020-12` (inclusive). Here's an example of a set of commands
that will set up your environment on the BB5:
```shell
module purge
module load archive/2020-12
module load python
python -m venv venv
. ./venv/bin/activate
python --version
```

We also recommend that you make sure that `pip` is up-to-date and that the
packages `wheel` and `setuptools` are installed:
```shell
pip install --upgrade pip wheel setuptools
```

### Install "Atlas Interpolation"
In order to access the data and the example scripts a local clone of this
repository is required. Run these commands to get it:
```shell
git clone https://github.com/BlueBrain/atlas-interpolation
cd atlas-interpolation
```

The "Atlas Interpolation" package can now be installed directly from the clone
we just created:
```shell
pip install '.[data, optical]'
```

## Data
The data for this project is managed by the [DVC tool](https://dvc.org/) and all
related files are located in the `data` directory. The DVC tool has already been
installed together with the "Atlas Interpolation" package. Every time you need
to run a DVC command (`dvc ...`) make sure to change to the `data` directory
first (`cd data`).

### Remote Storage Access
We have already prepared all the data, but it is located on a remote storage
that is only accessible to people within the Blue Brain Project who have
access permissions to project `proj101`. If you're unsure you can test your
permissions with the following command:
```shell
ssh bbpv1.bbp.epfl.ch \
"ls /gpfs/bbp.cscs.ch/data/project/proj101/dvc_remotes"
```
Possible outcomes:
```shell
# Access OK
atlas_annotation
atlas_interpolation

# Access denied
ls: cannot open directory [...]: Permission denied
```
Depending on whether you have access to the remote storage in the following
sections you will either pull the data from the remote (`dvc pull`) or download
the input data manually and re-run the data processing pipelines to reproduce
the output data (`dvc repro`).

If you work on the BB5 and have access to the remote storage then run the
following command to short-circuit the remote access (because the remote is
located on the BB5 itself):
```shell
dvc remote add --local gpfs_proj101 \
/gpfs/bbp.cscs.ch/data/project/proj101/dvc_remotes/atlas_interpolation
```

### Model Checkpoints
Much of the functionality of "Atlas Interpolation" relies on pre-trained deep
learning models. The model checkpoints that need to be loaded are part of the
data.

If you have access to the remote storage (see above) you can pull all model
checkpoints from the remote:
```shell
cd data
dvc pull checkpoints/rife.dvc
dvc pull checkpoints/cain.dvc
dvc pull checkpoints/maskflownet.params.dvc
dvc pull checkpoints/RAFT.dvc
```

If you don't have access to the remote you need to download the checkpoint files
by hand and put the downloaded data into the `data/checkpoints` folder. You
may not need all the checkpoints depending on the examples you want to run. Here
are the instructions for the four models we use: RIFE, CAIN, MaskFlowNet, and
RAFT:
* **RIFE**: download the checkpoint from a shared Google Drive folder by following
  [this link](https://drive.google.com/file/d/1wsQIhHZ3Eg4_AfCXItFKqqyDMB4NS0Yd/view?usp=sharing).
  Unzip the contents of the downloaded file into `data/checkpoints/rife`.
  [[ref]](https://github.com/hzwer/arXiv2020-RIFE/tree/6ff174584737a9aa27cd9654443a8a79c76799c9#usage)
* **CAIN**: download the checkpoint from a shared Dropbox folder by following
  [this link](https://www.dropbox.com/s/y1xf46m2cbwk7yf/pretrained_cain.pth?dl=0).
  Move the downloaded file to `data/checkpoints/cain`.
  [[ref]](https://github.com/myungsub/CAIN/tree/2e727d2a07d3f1061f17e2edaa47a7fb3f7e62c5#interpolating-with-custom-video)
* **MaskFlowNet**: download the checkpoint directly from GitHub by following
  [this link](https://github.com/microsoft/MaskFlownet/raw/5cba12772e2201f0d1c1e27161d224e585334571/weights/8caNov12-1532_300000.params).
  Rename the file to `maskflownet.params` and move it to `data/checkpoints`.
  [[ref]](https://github.com/microsoft/MaskFlownet/raw/5cba12772e2201f0d1c1e27161d224e585334571/weights)
* **RAFT**: downlaod the checkpoint files from a shared Dropbox folder by following
  [this link](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing).
  Move all downloaded `.pth` files to the `data/checkpoints/RAFT/models` folder.
  [[ref]](https://github.com/princeton-vl/RAFT/tree/224320502d66c356d88e6c712f38129e60661e80#demos)

If you downloaded all checkpoints or pulled them from the remote you should
have the following files:
```text
data
└── checkpoints
    ├── RAFT
    │   ├── models
    │   │   ├── raft-chairs.pth
    │   │   ├── raft-kitti.pth
    │   │   ├── raft-sintel.pth
    │   │   ├── raft-small.pth
    │   │   └── raft-things.pth
    ├── cain
    │   └── pretrained_cain.pth
    ├── maskflownet.params
    └── rife
        ├── contextnet.pkl
        ├── flownet.pkl
        └── unet.pkl
```

### Section Images and Datasets
The purpose of the "Atlas Interpolation" package is to interpolate missing
section images within section image datasets. This section explains how to
obtain these data.

Remember that if you don't have access to the remote storage (see above) you'll
need to use the `dvc repro` commands that download/process the data live. If
you do have access, you'll use `dvc pull` instead, which is faster.

Normally it's not necessary to get all data. Due to its size it may take a lot
of disk space as well as time to download and pre-process. If you still decide
to do so you can by running `dvc repro` or `dvc pull` without any parameters.

Specific examples only require specific data. You can use DVC to list all data
pipeline stages to find out which stage produces the data you're interested in.
To list all data pipeline stages run:
```shell
cd data
dvc stage list
```
If, for example, you need data located in `data/aligned/coronal/Gad1`, then
according to the output of command above the relevant stage is named
`align@Gad1`. Therefore, you only need to run this stage to get the necessary
data (replace `repro` by `pull` if you can access the remote storage):
```shell
dvc repro align@Gad1
```

### New ISH datasets (advanced, optional)
If you're familiar with the AIBS data that we're using and would like to add
new ISH gene expressions that are not yet available as one of our pipeline
stages (check the output of `dvc stage list`) then follow the following
instructions.

1. Edit the file `data/dvc.yaml` and add the new gene name to the lists in the
   `stages:download_dataset:foreach` and `stages:align:foreach` sections.
2. Run the data downloading and processing pipelines (replace `NEW_GENE` by the
   real gene name that you used in `data/dvc.yaml`):
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

dvc pull checkpoints/rife.dvc  # RIFE model
#dvc pull checkpoints/cain.dvc # CAIN model
```

If you are not able to pull:
- For RIFE: please follow instructions from https://github.com/hzwer/arXiv2020-RIFE#cli-usage
to download the model
- For CAIN: please follow instructions from https://github.com/myungsub/CAIN#usage 
to download the model

#### Example Code

Please be at the root folder of the project or change the `checkpoint_path`
to run the example code below properly.

```python
import numpy as np

from atlinter.vendor.rife.RIFE_HD import Model as RifeModel
from atlinter.vendor.rife.RIFE_HD import device as rife_device
from atlinter.pair_interpolation import PairInterpolate, RIFEPairInterpolationModel

# Instantiate the Pair Interpolation model (in this case: RIFE)
checkpoint_path = "data/checkpoints/rife/" # Please change, if needed
rife_model = RifeModel()
rife_model.load_model(checkpoint_path, -1)
rife_model.eval()
rife_interpolation_model = RIFEPairInterpolationModel(rife_model, rife_device)

# Predict middle image between img1 and img2
img1 = np.random.rand(100, 200, 3) # replace by real section image
img2 = np.random.rand(100, 200, 3) # replace by real section image
preimg1, preimg2 = rife_interpolation_model.before_interpolation(img1=img1, img2=img2)
img_middle = rife_interpolation_model.interpolate(img1=preimg1, img2=preimg2)
img_middle = rife_interpolation_model.after_interpolation(img_middle)

# If you want to predict several images between img1 and img2
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

dvc pull checkpoints/maskflownet.params.dvc # MaskFlowNet model
#dvc pull checkpoints/RAFT.dvc              # RAFT model
```

If you are not able to pull:
- For MaskFlowNet: please go to https://github.com/microsoft/MaskFlownet/tree/master/weights
and download `8caNov12-1532_300000.params` file.
- For RAFT: please follow the instructions from https://github.com/princeton-vl/RAFT#demos
to download the model.

#### Example Code

Please be at the root folder of the project or change the `checkpoint_path`
to run the example code below properly.

```python
import numpy as np

from atlinter.optical_flow import MaskFlowNet

# Instantiate the Optical Flow model (in this case: MaskFlowNet)
checkpoint_path = "data/checkpoints/maskflownet.params" # Please change, if needed
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
If it is not the case, please download it:
```shell
cd data

dvc pull download_dataset@Vip
```
If you do not have access to `proj101`, please replace `dvc pull` by `dvc repro`.
This might take some time.

#### Example Code

Please be at the root folder of the project or change the different paths
to run the example code below properly. The code might take some time, especially
the prediction of the entire volume (last line of the code), an environement
with GPUs could speed up the runtime.

```python
import json

import numpy as np

from atlinter.data import GeneDataset
from atlinter.pair_interpolation import GeneInterpolate, RIFEPairInterpolationModel
from atlinter.vendor.rife.RIFE_HD import Model as RifeModel
from atlinter.vendor.rife.RIFE_HD import device as rife_device

# 1.  Prepare dataset
# 1.a Load gene and section numbers
data_path = "data/sagittal/Vip/1102.npy"  # Change the path if needed
data_json = "data/sagittal/Vip/1102.json" # Change the path if needed
gene = np.load(data_path)
with open(data_json) as f:
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
checkpoint_path = "data/checkpoints/rife/"  # Change the path if needed
rife_model = RifeModel()
rife_model.load_model(checkpoint_path, -1)
rife_model.eval()
rife_interpolation_model = RIFEPairInterpolationModel(rife_model, rife_device)

# 3. Instantiate GeneInterpolate and predict specific slice or entire volume
gene_interpolate = GeneInterpolate(gene_dataset, rife_interpolation_model)
predicted_slice = gene_interpolate.predict_slice(10)
predicted_volume = gene_interpolate.predict_volume()  # This might take some time.
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
