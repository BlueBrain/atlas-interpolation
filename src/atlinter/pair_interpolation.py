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
"""Volume interpolation based on pairwise interpolation between slices."""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from math import ceil, log2

import numpy as np
import torch
from torchvision.transforms import ToTensor

from atlinter.data import GeneDataset
from atlinter.utils import find_closest


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
        image_shape = img1.shape
        if len(image_shape) == 3 and image_shape[-1] == 3:
            rgb = True
            image_shape = image_shape[:-1]
        else:
            rgb = False

        self.shape = np.array(image_shape)
        pad_x, pad_y = ((self.shape - 1) // 32 + 1) * 32 - self.shape

        if rgb:
            img1 = np.pad(img1, ((0, pad_x), (0, pad_y), (0, 0)))
            img2 = np.pad(img2, ((0, pad_x), (0, pad_y), (0, 0)))
        else:
            img1 = np.pad(img1, ((0, pad_x), (0, pad_y)))
            img2 = np.pad(img2, ((0, pad_x), (0, pad_y)))

        return img1, img2

    def interpolate(self, img1, img2):
        """Interpolate two images using RIFE.

        Note: img1 and img2 needs to have the same shape.
        If img1, img2 are grayscale, the dimension should be (height, width).
        If img1, img2 are RGB image, the dimension should be (height, width, 3).

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
        # Add batch and RGB dimensions (if not already), set device
        if len(img1.shape) == 2:
            rgb = False
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
        else:
            rgb = True
            img1 = np.transpose(img1, (2, 0, 1))[np.newaxis]
            img2 = np.transpose(img2, (2, 0, 1))[np.newaxis]
            img1 = torch.tensor(img1, dtype=torch.float32).to(self.rife_device)
            img2 = torch.tensor(img2, dtype=torch.float32).to(self.rife_device)

        # The actual interpolation
        img_mid = self.rife_model.inference(img1, img2).detach().cpu().numpy()

        img_mid = img_mid.squeeze()
        if rgb:
            # Put the RGB channel at the end
            img_mid = np.transpose(img_mid, (1, 2, 0))
        else:
            # Average out the RGB dimension
            img_mid = img_mid.mean(axis=0)

        return img_mid

    def after_interpolation(self, interpolated_images):
        """Undo the padding added in `before_interpolation`.

        Parameters
        ----------
        interpolated_images : np.ndarray
            The stacked interpolated images.
            If input images are grayscale,
            the dimension should be (n_img, height, width) or (height, width).
            If input images are RGB image,
            the dimension should be (n_img, height, width, 3) or (height, width, 3).

        Returns
        -------
        np.ndarray
            The stacked interpolated images with padding removed.
        """
        # No n_img dimension: (height, width) or (height, width, 3)
        if len(interpolated_images.shape) == 2 or (
            len(interpolated_images.shape) == 3 and interpolated_images.shape[-1] == 3
        ):
            return interpolated_images[: self.shape[0], : self.shape[1]]
        # n_img dimension: (n_img, height, width) or (n_img, height, width, 3)
        else:
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

        Note: img1 and img2 needs to have the same shape.
        If img1, img2 are grayscale, the dimension should be (height, width).
        If img1, img2 are RGB image, the dimension should be (height, width, 3).

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
        if len(img1.shape) == 2:
            rgb = False
            img1 = self.to_tensor(img1).repeat((1, 3, 1, 1))
            img2 = self.to_tensor(img2).repeat((1, 3, 1, 1))
        else:
            rgb = True
            img1 = self.to_tensor(np.transpose(img1, (2, 0, 1)))[None]
            img2 = self.to_tensor(np.transpose(img2, (2, 0, 1)))[None]

        # The actual interpolation
        img_mid, _ = self.cain_model(img1, img2)
        img_mid = img_mid.detach().cpu().numpy()

        img_mid = img_mid.squeeze()
        if rgb:
            # Put the RGB channel at the end
            img_mid = np.transpose(img_mid, (1, 2, 0))
        else:
            # Average out the RGB dimension
            img_mid = img_mid.mean(axis=0)

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


class GeneInterpolate:
    """Interpolation of a gene dataset.

    Parameters
    ----------
    gene_data : GeneData
        Gene Dataset to interpolate. It contains a `volume` of reference shape
        with all known places located at the right place and a `metadata` dictionary
        containing information about the axis of the dataset and the section numbers.

    model : PairInterpolationModel
        Pair-interpolation model.
    """

    def __init__(
        self,
        gene_data: GeneDataset,
        model: PairInterpolationModel,
    ):
        self.gene_data = gene_data
        self.model = model

        self.axis = self.gene_data.axis
        self.gene_volume = self.gene_data.volume.copy()
        # If sagittal axis, put the sagittal dimension first
        if self.axis == "sagittal":
            self.gene_volume = np.moveaxis(self.gene_volume, 2, 0)

    def get_interpolation(
        self, left: int, right: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Compute the interpolation for a pair of images.

        Parameters
        ----------
        left
            Slice number of the left image to consider.
        right
            Slice number of the right image to consider.

        Returns
        -------
        interpolated_images : np.array or None
            Interpolated image for the given pair of images.
            Array of shape (N, dim1, dim2, 3) with N the number of
            interpolated images.
        predicted_section_numbers : np.array or None
            Slice value of the predicted images.
            Array of shape (N, 1) with N the number of interpolated images.
        """
        diff = right - left
        if diff == 0:
            return None, None

        n_repeat = self.get_n_repeat(diff)

        pair_interpolate = PairInterpolate(n_repeat=n_repeat)
        interpolated_images = pair_interpolate(
            self.gene_volume[left], self.gene_volume[right], self.model
        )
        predicted_section_numbers = self.get_predicted_section_numbers(
            left, right, n_repeat
        )
        return interpolated_images, predicted_section_numbers

    def get_all_interpolation(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute pair interpolation for the entire volume.

        Returns
        -------
        all_interpolated_images : np.array
            Interpolated image for the entire volume.
            Array of shape (N, dim1, dim2, 3) with N the number of
            interpolated images.
        all_predicted_section_numbers : np.array
            Slice value of the predicted images.
            Array of shape (N, 1) with N the number of interpolated images.
        """
        # TODO: Try to change the implementation of the prediction so that
        # we do not predict slices that are not needed.
        known_slices = sorted(self.gene_data.known_slices)

        all_interpolated_images = []
        all_predicted_section_numbers = []
        for i in range(len(known_slices) - 1):
            left, right = known_slices[i], known_slices[i + 1]
            (
                interpolated_images,
                predicted_section_numbers,
            ) = self.get_interpolation(left, right)
            if interpolated_images is None:
                continue
            all_interpolated_images.append(interpolated_images)
            all_predicted_section_numbers.append(predicted_section_numbers)

        all_interpolated_images = np.concatenate(all_interpolated_images)
        all_predicted_section_numbers = np.concatenate(all_predicted_section_numbers)
        return all_interpolated_images, all_predicted_section_numbers

    def predict_slice(self, slice_number: int) -> np.ndarray:
        """Predict one gene slice.

        Parameters
        ----------
        slice_number
            Slice section to predict.

        Returns
        -------
        np.ndarray
            Predicted gene slice. Array of shape (dim1, dim2, 3)
            being (528, 320) for sagittal dataset and
            (320, 456) for coronal dataset.
        """
        left, right = self.gene_data.get_surrounding_slices(slice_number)

        if left is None:
            return self.gene_volume[right]
        elif right is None:
            return self.gene_volume[left]
        else:
            interpolated_images, predicted_section_numbers = self.get_interpolation(
                left, right
            )
            index = find_closest(slice_number, predicted_section_numbers)[0]
            return interpolated_images[index]

    def predict_volume(self) -> np.ndarray:
        """Predict entire volume with known gene slices.

        This function might be slow.
        """
        volume_shape = self.gene_data.volume_shape
        volume = np.zeros(volume_shape)

        if self.gene_data.axis == "sagittal":
            volume = np.moveaxis(volume, 2, 0)
        # Get all the predictions
        (
            all_interpolated_images,
            all_predicted_section_numbers,
        ) = self.get_all_interpolation()

        min_slice_number = min(self.gene_data.known_slices)
        max_slice_number = max(self.gene_data.known_slices)
        end = volume_shape[0] if self.gene_data.axis == "coronal" else volume_shape[2]

        # Populate the volume
        for slice_number in range(end):
            # If the slice is known, just copy the gene.
            if slice_number in self.gene_data.known_slices:
                volume[slice_number] = self.gene_volume[slice_number]
            # If the slice section is smaller than all known slice
            # We copy-paste the smallest known slice.
            elif slice_number < min_slice_number:
                volume[slice_number] = self.gene_volume[min_slice_number]
            # If the slice section is bigger than all known slice
            # We copy-paste the biggest known slice.
            elif slice_number > max_slice_number:
                volume[slice_number] = self.gene_volume[max_slice_number]
            # If the slice is surrounded by two known slice.
            # Determine the prediction closest to the slice section.
            else:
                index = find_closest(slice_number, all_predicted_section_numbers)[0]
                volume[slice_number] = all_interpolated_images[index]

        if self.gene_data.axis == "sagittal":
            volume = np.moveaxis(volume, 0, 2)

        return volume

    @staticmethod
    def get_n_repeat(diff: int) -> int:
        """Determine the number of repetitions to compute."""
        if diff <= 0:
            return 0
        n_repeat = ceil(log2(diff))
        return n_repeat

    @staticmethod
    def get_predicted_section_numbers(
        left: int, right: int, n_repeat: int
    ) -> np.ndarray:
        """Get slice values of predicted images."""
        n_steps = 2 ** n_repeat + 1
        predicted_section_numbers = np.linspace(left, right, n_steps)
        return predicted_section_numbers[1:-1]
