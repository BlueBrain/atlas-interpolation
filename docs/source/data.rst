Data
====
The data for this project is managed by the `DVC tool <https://dvc.org>`__ and
all related files are located in the ``data`` directory. The DVC tool has
already been installed together with the "Atlas Interpolation" package. Every
time you need to run a DVC command (``dvc ...``) make sure to change to the
``data`` directory first (``cd data``).

Remote Storage Access
---------------------
We have already prepared all the data, but it is located on a remote storage
that is only accessible to people within the Blue Brain Project who have
access permissions to project ``proj101``. If you're unsure you can test your
permissions with the following command:

.. code-block:: shell

    ssh bbpv1.bbp.epfl.ch \
    "ls /gpfs/bbp.cscs.ch/data/project/proj101/dvc_remotes"

Possible outcomes:

.. code-block:: shell

    # Access OK
    atlas_annotation
    atlas_interpolation

    # Access denied
    ls: cannot open directory [...]: Permission denied

Depending on whether you have access to the remote storage in the following
sections you will either pull the data from the remote (``dvc pull``) or
download the input data manually and re-run the data processing pipelines to
reproduce the output data (``dvc repro``).

If you work on the BB5 and have access to the remote storage then run the
following command to short-circuit the remote access (because the remote is
located on the BB5 itself):

.. code-block:: shell

    cd data
    dvc remote add --local gpfs_proj101 \
      /gpfs/bbp.cscs.ch/data/project/proj101/dvc_remotes/atlas_interpolation
    cd ..


Model Checkpoints
-----------------
Much of the functionality of "Atlas Interpolation" relies on pre-trained deep
learning models. The model checkpoints that need to be loaded are part of the
data.

If you have access to the remote storage (see above) you can pull all model
checkpoints from the remote:

.. code-block:: shell

    cd data
    dvc pull checkpoints/rife.dvc
    dvc pull checkpoints/cain.dvc
    dvc pull checkpoints/maskflownet.params.dvc
    dvc pull checkpoints/RAFT.dvc
    cd ..

If you don't have access to the remote you need to download the checkpoint files
by hand and put the downloaded data into the ``data/checkpoints`` folder. You
may not need all the checkpoints depending on the examples you want to run. Here
are the instructions for the four models we use: RIFE, CAIN, MaskFlowNet, and
RAFT:

* **RIFE**: download the checkpoint from a shared Google Drive folder by following
  `this link <https://drive.google.com/file/d/11l8zknO1V5hapv2-Ke4DG9mHyBomS0Fc/view?usp=sharing>`__.
  Unzip the contents of the downloaded file into ``data/checkpoints/rife``.
  `[ref] <https://github.com/hzwer/arXiv2020-RIFE/tree/2a1eafe27d5ff12eb31df96e47352fe30c18ac46#usage>`__
* **CAIN**: download the checkpoint from a shared Dropbox folder by following
  `this link <https://www.dropbox.com/s/y1xf46m2cbwk7yf/pretrained_cain.pth?dl=0>`__.
  Move the downloaded file to ```data/checkpoints/cain``.
  `[ref] <https://github.com/myungsub/CAIN/tree/2e727d2a07d3f1061f17e2edaa47a7fb3f7e62c5#interpolating-with-custom-video>`__
* **MaskFlowNet**: download the checkpoint directly from GitHub by following
  `this link <https://github.com/microsoft/MaskFlownet/raw/5cba12772e2201f0d1c1e27161d224e585334571/weights/8caNov12-1532_300000.params>`__.
  Rename the file to ``maskflownet.params`` and move it to ``data/checkpoints``.
  `[ref] <https://github.com/microsoft/MaskFlownet/raw/5cba12772e2201f0d1c1e27161d224e585334571/weights>`__
* **RAFT**: download the checkpoint files from a shared Dropbox folder by following
  `this link <https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing>`__.
  Move all downloaded ``.pth`` files to the ``data/checkpoints/RAFT/models`` folder.
  `[ref] <https://github.com/princeton-vl/RAFT/tree/224320502d66c356d88e6c712f38129e60661e80#demos>`__

If you downloaded all checkpoints or pulled them from the remote you should
have the following files:

.. code-block:: text

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


Section Images and Datasets
---------------------------
The purpose of the "Atlas Interpolation" package is to interpolate missing
section images within section image datasets. This section explains how to
obtain these data.

Remember that if you don't have access to the remote storage (see above) you'll
need to use the ``dvc repro`` commands that download/process the data live. If
you do have access, you'll use ``dvc pull`` instead, which is faster.

Normally it's not necessary to get all data. Due to its size it may take a lot
of disk space as well as time to download and pre-process. If you still decide
to do so you can by running ``dvc repro`` or ``dvc pull`` without any
parameters.

Specific examples only require specific data. You can use DVC to list all data
pipeline stages to find out which stage produces the data you're interested in.
To list all data pipeline stages run:

.. code-block:: shell

    cd data
    dvc stage list

If, for example, you need data located in ``data/aligned/coronal/Gad1``, then
according to the output of command above the relevant stage is named
``align@Gad1``. Therefore, you only need to run this stage to get the necessary
data (replace ``repro`` by ``pull`` if you can access the remote storage):

.. code-block:: shell

    dvc repro align@Gad1


New ISH datasets (advanced, optional)
-------------------------------------
If you're familiar with the AIBS data that we're using and would like to add
new ISH gene expressions that are not yet available as one of our pipeline
stages (check the output of ``dvc stage list``) then follow the following
instructions.

1. Edit the file ``data/dvc.yaml`` and add the new gene name to the lists in the
   ``stages:download_dataset:foreach`` and ``stages:align:foreach`` sections.
2. Run the data downloading and processing pipelines (replace ``NEW_GENE`` by
   the real gene name that you used in ``data/dvc.yaml``):

   .. code-block:: shell

      dvc repro download_dataset@NEW_GENE
      dvc repro align@NEW_GENE
