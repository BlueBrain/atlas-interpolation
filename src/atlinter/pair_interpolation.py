"""Volume interpolation based on pairwise interpolation between slices."""
import warnings
from abc import ABC, abstractmethod

import numpy as np
import torch
from torchvision.transforms import ToTensor


class PairInterpolationModel(ABC):
    """Base class for pair-interpolation models.

    Subclasses of this class implement an interpolation between two given
    images `img1` and `img2` to produce and intermediate image `img_mid`.

    This class and its subclasses are used by the PairInterpolate class,
    which applies a given interpolation model to concrete data.
    """

    def before_interpolation(self, img1, img2):
        """Run initialization and pre-processing steps before interpolation.

        Typical applications of this method are padding and cropping of
        input images to fit the model requirements, as well as initialisation
        of any internal state, should one be necessary.

        Parameters
        ----------
        img1 : np.ndarray
            The left image of shape (width, height)
        img2 : np.ndarray
            The right image of shape (width, height).

        Returns
        -------
        img1 : np.ndarray
            The pre-processed left image.
        img2 : np.ndarray
            The pre-processed right image.
        """
        return img1, img2

    @abstractmethod
    def interpolate(self, img1, img2):
        """Interpolate two images.

        In the typical setting the input images are going to be of the format
        as returned by the `before_interpolation`.

        Parameters
        ----------
        img1 : np.ndarray
            The left image.
        img2 : np.ndarray
            The right image.

        Returns
        -------
        img_mid : np.ndarray
            The interpolated image.
        """

    def after_interpolation(self, interpolated_images):
        """Run any post-processing after all interpolation is done.

        Typical applications are padding and cropping of the image stack,
        as well as any clean-up of the model state.

        Parameters
        ----------
        interpolated_images : np.ndarray
            The stacked interpolated images. The array will include the input
            images as the first and the last items respectively and will
            therefore have the shape (n_interpolated + 2, height, width)

        Returns
        -------
        np.ndarray
            The post-processed interpolated images.
        """
        return interpolated_images


class LinearPairInterpolationModel(PairInterpolationModel):
    """Linear pairwise interpolation.

    This is the simplest possible interpolation model where the middle
    image is the average of the left and right images.
    """

    def interpolate(self, img1, img2):
        """Interpolate two images using linear interpolation.

        Parameters
        ----------
        img1 : np.ndarray
            The left image.
        img2 : np.ndarray
            The right image.

        Returns
        -------
        img_mid : np.ndarray
            The interpolated image.
        """
        img_mid = np.mean([img1, img2], axis=0)

        return img_mid


class RIFEPairInterpolationModel(PairInterpolationModel):
    """Pairwise image interpolation using the RIFE model.

    The typical use is
    >>> from atlinter.vendor.rife.RIFE_HD import Model as RifeModel
    >>> from atlinter.vendor.rife.RIFE_HD import device as rife_device
    >>> rife_model = RifeModel()
    >>> rife_model.load_model("/path/to/train_log", -1)
    >>> rife_model.eval()
    >>> rife_interpolation_model = RIFEPairInterpolationModel(rife_model, rife_device)

    Parameters
    ----------
    rife_model : atlinter.vendor.rife.RIFE_HD.Model
        The RIFE model instance.
    rife_device : from atlinter.vendor.rife.RIFE_HD.device
        The RIFE device.
    """

    def __init__(self, rife_model, rife_device):
        # The behaviour of torch.nn.functional.interpolate has slightly changed,
        # which leads to this warning. It doesn't seem to have an impact on the
        # results, but if the authors of RIFE decide to update their code base
        # by either specifying the `recompute_scale_factor` parameter or by
        # some other means, then this warning filter should be removed.
        # TODO: check the RIFE code for updates and remove the filter if necessary.
        warnings.filterwarnings(
            "ignore",
            "The default behavior for interpolate/upsample with float scale_factor",
            UserWarning,
        )
        self.rife_model = rife_model
        self.rife_device = rife_device
        self.shape = (0, 0)

    def before_interpolation(self, img1, img2):
        """Pad input images to a multiple of 32 pixels.

        Parameters
        ----------
        img1 : np.ndarray
            The left image of shape.
        img2 : np.ndarray
            The right image of shape.

        Returns
        -------
        img1 : np.ndarray
            The padded left image.
        img2 : np.ndarray
            The padded right image.
        """
        self.shape = np.array(img1.shape)
        pad_x, pad_y = ((self.shape - 1) // 32 + 1) * 32 - self.shape
        img1 = np.pad(img1, ((0, pad_x), (0, pad_y)))
        img2 = np.pad(img2, ((0, pad_x), (0, pad_y)))

        return img1, img2

    def interpolate(self, img1, img2):
        """Interpolate two images using RIFE.

        Parameters
        ----------
        img1 : np.ndarray
            The left image.
        img2 : np.ndarray
            The right image.

        Returns
        -------
        img_mid : np.ndarray
            The interpolated image.
        """
        # Add batch and RGB dimensions, set device
        img1 = (
            torch.tensor(img1, dtype=torch.float32)
            .repeat((1, 3, 1, 1))
            .to(self.rife_device)
        )
        img2 = (
            torch.tensor(img2, dtype=torch.float32)
            .repeat((1, 3, 1, 1))
            .to(self.rife_device)
        )

        # The actual interpolation
        img_mid = self.rife_model.inference(img1, img2).detach().cpu().numpy()

        # Average out the RGB dimension
        img_mid = img_mid.squeeze().mean(axis=0)

        return img_mid

    def after_interpolation(self, interpolated_images):
        """Undo the padding added in `before_interpolation`.

        Parameters
        ----------
        interpolated_images : np.ndarray
            The stacked interpolated images.

        Returns
        -------
        np.ndarray
            The stacked interpolated images with padding removed.
        """
        return interpolated_images[:, : self.shape[0], : self.shape[1]]


class CAINPairInterpolationModel(PairInterpolationModel):
    """Pairwise image interpolation using the CAIN model.

    The typical use is
    >>> from atlinter.vendor.cain.cain import CAIN
    >>> device = "cuda" if torch.cuda.is_available else "cpu"
    >>> cain_model = torch.nn.DataParallel(CAIN()).to(device)
    >>> cain_checkpoint = torch.load("pretrained_cain.pth", map_location=device)
    >>> cain_model.load_state_dict(cain_checkpoint["state_dict"])
    >>> cain_interpolation_model = CAINPairInterpolationModel(cain_model)

    Parameters
    ----------
    cain_model : atlinter.vendor.cain.cain.CAIN or torch.nn.DataParallel
        The CAIN model instance.
    """

    def __init__(self, cain_model):
        self.cain_model = cain_model
        self.to_tensor = ToTensor()

    def interpolate(self, img1, img2):
        """Interpolate two images using CAIN.

        Parameters
        ----------
        img1 : np.ndarray
            The left image.
        img2 : np.ndarray
            The right image.

        Returns
        -------
        img_mid : np.ndarray
            The interpolated image.
        """
        # Add batch and RGB dimensions
        img1 = self.to_tensor(img1).repeat((1, 3, 1, 1))
        img2 = self.to_tensor(img2).repeat((1, 3, 1, 1))

        # The actual interpolation
        img_mid, _ = self.cain_model(img1, img2)
        img_mid = img_mid.detach().cpu().numpy()

        # Average out the RGB dimension
        img_mid = img_mid.squeeze().mean(axis=0)

        return img_mid


class AntsPairInterpolationModel(PairInterpolationModel):
    """Pairwise image interpolation using AntsPy registration.

    Typical use is
    >>> from atlannot.ants import register, transform
    >>> ants_interpolation_model = AntsPairInterpolationModel(register, transform)

    Parameters
    ----------
    register_fn : atlannot.ants.register
        The AntsPy registration function
    transform_fn : atlannot.ants.transform
        The AntsPy transformation function
    """

    def __init__(self, register_fn, transform_fn):
        self.register_fn = register_fn
        self.transform_fn = transform_fn

    def interpolate(self, img1, img2):
        """Interpolate two images using AntsPy registration.

        Parameters
        ----------
        img1 : np.ndarray
            The left image.
        img2 : np.ndarray
            The right image.

        Returns
        -------
        img_mid : np.ndarray
            The interpolated image.
        """
        # Ensure the correct d-type
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        # The actual interpolation
        nii_data = self.register_fn(fixed=img2, moving=img1)
        img_mid = self.transform_fn(img1, nii_data / 2)

        return img_mid


class PairInterpolate:
    """Runner for pairwise interpolation using different models.

    Parameters
    ----------
    n_repeat : int (optional)
        The number of times the interpolation should be iterated. For each
        iteration an interpolated image is inserted between each pair of
        images from the previous iteration. Therefore n_{i+1} = n_i + (n_i + 1).
        For example, for n_repeat=3 the progression of the number of images
        will be the following: input = 0 -> 1 -> 3 -> 7
    """

    def __init__(self, n_repeat=1):
        self.n_repeat = n_repeat

    def repeat(self, n_repeat):
        """Set the number of interpolation iterations.

        Parameters
        ----------
        n_repeat : int
            The new number of interpolation iterations. See `__init__` for more
            details.
        """
        self.n_repeat = n_repeat

        return self

    def __call__(self, img1, img2, model: PairInterpolationModel):
        """Run the interpolation with the given interpolation model.

        Parameters
        ----------
        img1 : np.ndarray
            The left input image.
        img2 : np.ndarray
            The right input image.
        model : PairInterpolationModel
            The interpolation model.

        Returns
        -------
        interpolated_images : np.ndarray
            A stack of interpolation images. The input images are not included
            in this stack.
        """
        img1, img2 = model.before_interpolation(img1, img2)
        interpolated_images = self._repeated_interpolation(
            img1, img2, model, self.n_repeat
        )
        interpolated_images = np.stack(interpolated_images)
        interpolated_images = model.after_interpolation(interpolated_images)

        return interpolated_images

    def _repeated_interpolation(self, img1, img2, model, n_repeat):
        # End of recursion
        if n_repeat <= 0:
            return []

        # Recursion step
        img_mid = model.interpolate(img1, img2)
        left_images = self._repeated_interpolation(img1, img_mid, model, n_repeat - 1)
        right_images = self._repeated_interpolation(img_mid, img2, model, n_repeat - 1)

        return [*left_images, img_mid, *right_images]
