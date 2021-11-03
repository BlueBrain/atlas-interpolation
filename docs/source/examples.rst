Examples
========
In this section we showcase several typical use-cases of "Atlas Interpolation":

* Use pair interpolation to predict an intermediate image between two given
  images
* Predict optical flow between any pair of images and use it to morph a third
  image
* In a gene expression volume predict missing slices and reconstruct the whole
  volume

Note that all models accept both RGB images (``shape=(height, width, 3)``)
and grayscale images (``shape=(height, width)``).

Pair Interpolation
------------------
The only data you need for this example is the RIFE model checkpoint. Follow
the instructions in the corresponding section above to get it. If you have
access to the remote data storage it's enough to run the following commands:

.. code-block:: shell

    cd data
    dvc pull checkpoints/rife.dvc
    cd ..

In this example we start with a pair of images ``img1`` and ``img2`` (randomly
generated for example's sake). First use the RIFE model to interpolate between
them in a manual way and find the image in-between (``img_middle``). Then we
demonstrate the use of the ``PairInterpolate`` class that streamlines the
interpolation procedure. Starting with the same pair of images we iterate the
interpolation three times to produce a stack of seven interpolated images
(``interpolated_imgs``).

.. code-block:: python

    import numpy as np

    from atlinter.vendor.rife.RIFE_HD import Model as RifeModel
    from atlinter.vendor.rife.RIFE_HD import device as rife_device
    from atlinter.pair_interpolation import PairInterpolate, RIFEPairInterpolationModel

    # Get the input images
    img1 = np.random.rand(100, 200, 3) # replace by real section image
    img2 = np.random.rand(100, 200, 3) # replace by real section image

    # Get the RIFE interpolation model
    checkpoint_path = "data/checkpoints/rife/" # Please change, if needed
    rife_model = RifeModel()
    rife_model.load_model(checkpoint_path, -1)
    rife_model.eval()
    interpolation_model = RIFEPairInterpolationModel(rife_model, rife_device)

    # Manually predict middle image between img1 and img2
    preimg1, preimg2 = interpolation_model.before_interpolation(img1=img1, img2=img2)
    img_middle = interpolation_model.interpolate(img1=preimg1, img2=preimg2)
    img_middle = interpolation_model.after_interpolation(img_middle)
    print(img_middle.shape)

    # Streamline the interpolation using PairInterpolate and predict a stack
    # of 7 intermediate images
    interpolator = PairInterpolate(n_repeat=3)
    interpolated_imgs = interpolator(img1, img2, interpolation_model)
    print(interpolated_imgs.shape)


Optical Flow Models
-------------------
The only data you need for this example is the MaskFlowNet model checkpoint.
Follow the instructions in the corresponding section above to get it. If you
have access to the remote data storage it's enough to run the following
commands:

.. code-block:: shell

    cd data
    dvc pull checkpoints/maskflownet.params.dvc
    cd ..

This example demonstrates how an optical flow model can be used to compute the
optical flow between a pair of images. It can then be used to warp a third
image. The images in this example are randomly generated. In a realistic setting
they should be replaced by real images.

.. code-block:: python

    import numpy as np

    from atlinter.optical_flow import MaskFlowNet

    # Instantiate an optical flow model (in this case: MaskFlowNet)
    checkpoint_path = "data/checkpoints/maskflownet.params"
    net = MaskFlowNet(checkpoint_path)

    # Prepare random images. Should be replaced by real section images
    img1 = np.random.rand(100, 200, 3)
    img2 = np.random.rand(100, 200, 3)
    img3 = np.random.rand(100, 200, 3)

    # Predict the optical flow between img1 and img2
    img1, img2 = net.preprocess_images(img1=img1, img2=img2)
    predicted_flow = net.predict_flow(img1=img1, img2=img2)

    # Warp a third image using the optical flow
    predicted_img = net.warp_image(predicted_flow, img3)
    print(predicted_img.shape)


Predict an Entire Gene Volume (Longer Runtime)
----------------------------------------------
The data you need for this example are the RIFE model checkpoint and the Vip
gene expression dataset. To get the RIFE checkpoint follow the instruction in
the corresponding section above. If you have access to the remote data storage
it's enough to run the following commands:

.. code-block:: shell

    cd data
    dvc pull checkpoints/rife.dvc
    cd ..

As described in the data section above, there are two ways of getting the Vip
gene expression dataset. If you have access to the remote data storage you can
pull it from there:

.. code-block:: shell

    cd data
    dvc pull download_dataset@Vip
    cd ..

If you don't have access then you can re-download it. This should always work,
but may take several minutes:

.. code-block:: shell

    cd data
    dvc repro download_dataset@Vip
    cd ..

In this example with start with a gene expression volume that has missing
section images. First we load the image data and the metadata from disk and
wrap it into a ``GeneDataset`` class. Then we instantiate the RIFE deep learning
model that will be used for interpolation. We use this model to first predict a
single slice in the volume, then we reconstruct the whole volume by predicting
all intermediate slices. Note that this last step is computation-heavy and might
therefore take some time.

.. code-block:: python

    import json

    import numpy as np

    from atlinter.data import GeneDataset
    from atlinter.pair_interpolation import GeneInterpolate, RIFEPairInterpolationModel
    from atlinter.vendor.rife.RIFE_HD import Model as RifeModel
    from atlinter.vendor.rife.RIFE_HD import device as rife_device

    # Load the gene expression dataset from disk
    data_path = "data/sagittal/Vip/1102.npy"  # Change the path if needed
    data_json = "data/sagittal/Vip/1102.json" # Change the path if needed
    section_images = np.load(data_path)
    with open(data_json) as fh:
        metadata = json.load(fh)

    section_numbers = [int(s) for s in metadata["section_numbers"]]
    axis = metadata["axis"]

    # Wrap the data into a GeneDataset class
    gene_dataset = GeneDataset(
      section_images,
      section_numbers,
      volume_shape=(528, 320, 456, 3),
      axis=axis,
    )

    # Load the RIFE deep learning model that will be used for interpolation
    checkpoint_path = "data/checkpoints/rife"
    rife_model = RifeModel()
    rife_model.load_model(checkpoint_path, -1)
    rife_model.eval()
    rife_interpolation_model = RIFEPairInterpolationModel(rife_model, rife_device)

    # Create a gene interpolator
    gene_interpolate = GeneInterpolate(gene_dataset, rife_interpolation_model)

    # Predict a single section image
    predicted_slice = gene_interpolate.predict_slice(10)
    print(predicted_slice.shape)

    # Reconstruct the whole volume. This might take some time.
    predicted_volume = gene_interpolate.predict_volume()
    print(predicted_volume.shape)
