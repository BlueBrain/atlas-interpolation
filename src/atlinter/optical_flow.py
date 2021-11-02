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
"""Optical flow computation based on pair of images."""
import argparse
import logging
from abc import ABC, abstractmethod

import mxnet as mx
import numpy as np
import torch
from scipy import ndimage

from atlinter.data import GeneDataset

logger = logging.getLogger(__name__)


class OpticalFlow(ABC):
    """Class representing optical flow model."""

    def preprocess_images(self, img1, img2):
        """Preprocess image.

        Parameters
        ----------
        img1 : np.ndarray
            The left image.
        img2 : np.ndarray
            The right image.

        Returns
        -------
        img1 : np.ndarray
            The pre-processed left image.
        img2 : np.ndarray
            The pre-processed right image.
        """
        return img1, img2

    @abstractmethod
    def predict_flow(self, img1, img2):
        """Compute optical flow between two images.

        Parameters
        ----------
        img1 : np.ndarray
            The left image.
        img2 : np.ndarray
            The right image.

        Returns
        -------
        flow : np.ndarray
            The optical flow of shape ``(*image.shape, 2)``.
        """

    @classmethod
    def warp_image(cls, flow, img2, order=1):
        """Warp image with the predicted flow.

        Parameters
        ----------
        flow : np.ndarray
            The predicted optical flow of shape ``(*image.shape, 2)``.
        img2 : np.ndarray
            The right image.
        order : int
            The interpolation order. 0 = nearest neighbour, 1 = linear,
            2 = cubic, etc.

        Returns
        -------
        warped : np.ndarray
            The warped image.
        """
        if flow.shape[1:3] != img2.shape[0:2]:
            raise ValueError("The flow shape and the image shape should be consistent.")

        if img2.ndim == 2:
            # greyscale
            ni, nj = img2.shape
            ij = np.mgrid[:ni, :nj]
            warped = ndimage.map_coordinates(img2, ij - flow, order=order)
        elif img2.ndim == 3 and img2.shape[-1] == 3:
            # RGB, warp each channel separately
            warped = np.stack(
                [
                    cls.warp_image(flow, img2[..., 0], order=order),
                    cls.warp_image(flow, img2[..., 1], order=order),
                    cls.warp_image(flow, img2[..., 2], order=order),
                ],
                axis=-1,
            )
        else:
            raise ValueError(f"Invalid image shape: {img2.shape}")

        return warped


class MaskFlowNet(OpticalFlow):
    """MaskFlowNet model for optical flow computation.

    The typical use is

    >>> from atlinter.optical_flow import MaskFlowNet
    >>> checkpoint = "data/checkpoints/maskflownet.params"
    >>> # Make sure the checkpoint exists and uncomment the following line
    >>> # net = MaskFlowNet(checkpoint)

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint of the model. See references for possibilities.
    gpu_device : str
        Device to load optical flow model.
    """

    from atlinter.vendor.MaskFlowNet import network, predict_new_data

    def __init__(self, checkpoint_path, gpu_device=""):
        config_file = {
            "network": {"class": "MaskFlownet"},
            "optimizer": {
                "learning_rate": [
                    [300000, 0.0001],
                    [500000, 5e-05],
                    [600000, 2.5e-05],
                    [700000, 1.25e-05],
                    [800000, 6.25e-06],
                ],
                "wd": 0.0004,
            },
        }
        config = self.network.config.Reader(config_file)

        if gpu_device == "":
            ctx = [mx.cpu()]
        else:
            ctx = [mx.gpu(int(gpu_id)) for gpu_id in gpu_device.split(",")]

        pipe = self.network.get_pipeline("MaskFlownet", ctx=ctx, config=config)
        self.pipe = self.predict_new_data.load_checkpoint(pipe, config, checkpoint_path)

    def preprocess_images(self, img1, img2):
        """Preprocess image.

        Parameters
        ----------
        img1 : np.ndarray
            The left image.
        img2 : np.ndarray
            The right image.

        Returns
        -------
        img1 : np.ndarray
            The pre-processed left image.
        img2 : np.ndarray
            The pre-processed right image.
        """
        if img1.shape != img2.shape:
            raise ValueError("The two images have not the same shape!")

        if np.max(img1) <= 1:
            img1 = (img1 * 255).astype(np.uint8)
            img2 = (img2 * 255).astype(np.uint8)

        if len(img1.shape) == 2:
            img1 = np.repeat(img1[..., np.newaxis], 3, axis=2)
            img2 = np.repeat(img2[..., np.newaxis], 3, axis=2)

        return img1, img2

    def predict_flow(self, img1, img2):
        """Compute optical flow between two images.

        Parameters
        ----------
        img1 : np.ndarray
            The left image.
        img2 : np.ndarray
            The right image.

        Returns
        -------
        flow : np.ndarray
            The optical flow.
        """
        flow, _, _ = self.predict_new_data.predict_image_pair_flow(
            img1, img2, self.pipe
        )
        return np.transpose(flow, (2, 0, 1))


class RAFTNet(OpticalFlow):
    """RAFT model for optical flow computation.

    The typical use is

    >>> from atlinter.optical_flow import RAFTNet
    >>> checkpoint = "data/checkpoints/RAFT/models/raft-things.pth"
    >>> # Make sure the checkpoint exists and uncomment the following line
    >>> # net = RAFTNet(checkpoint)

    Parameters
    ----------
    path : str
        Path to the RAFT model.
    device : {'cpu', 'cuda'}
        Device to load optical flow model.
    """

    from atlinter.vendor.RAFT.raft import RAFT

    def __init__(self, path, device="cpu"):
        self.device = device
        args = self.initialize_namespace()
        self.model = torch.nn.DataParallel(self.RAFT(args))
        self.model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        self.model = self.model.module
        if self.device == "cuda":
            self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def initialize_namespace():
        """Initialize namespace needed for RAFT initialization.

        Returns
        -------
        args : argparse.Namespace
            Arguments needed to instantiate RAFT model.
        """
        namespace = argparse.Namespace()
        namespace.small = False
        namespace.mixed_precision = False
        return namespace

    def preprocess_images(self, img1, img2):
        """Preprocess image.

        Parameters
        ----------
        img1 : np.ndarray
            The left image.
        img2 : np.ndarray
            The right image.

        Returns
        -------
        img1 : np.ndarray
            The pre-processed left image.
        img2 : np.ndarray
            The pre-processed right image.
        """
        if img1.shape != img2.shape:
            raise ValueError("The two images have not the same shape!")

        # The image dimensions need to be divisible by 8
        self.shape = np.array(img1.shape[:2])
        pad_x, pad_y = ((self.shape - 1) // 8 + 1) * 8 - self.shape

        if np.max(img1) <= 1:
            img1 = (img1 * 255).astype(np.uint8)
            img2 = (img2 * 255).astype(np.uint8)

        if len(img1.shape) == 2:
            img1 = np.pad(img1, ((0, pad_x), (0, pad_y)))
            img2 = np.pad(img2, ((0, pad_x), (0, pad_y)))

            img1 = np.repeat(img1[np.newaxis], 3, axis=0)
            img2 = np.repeat(img2[np.newaxis], 3, axis=0)
        else:
            img1 = np.pad(img1, ((0, pad_x), (0, pad_y), (0, 0)))
            img2 = np.pad(img2, ((0, pad_x), (0, pad_y), (0, 0)))

            img1 = np.transpose(img1, (2, 0, 1))
            img2 = np.transpose(img2, (2, 0, 1))

        img1 = img1[np.newaxis]
        img2 = img2[np.newaxis]

        return img1, img2

    def predict_flow(self, img1, img2):
        """Compute optical flow between two images.

        Parameters
        ----------
        img1 : np.ndarray
            The left image.
        img2 : np.ndarray
            The right image.

        Returns
        -------
        flow : np.ndarray
            The optical flow.
        """
        img1 = torch.from_numpy(img1).float().to(self.device)
        img2 = torch.from_numpy(img2).float().to(self.device)

        _, flow_up = self.model(img1, img2, iters=20, test_mode=True)
        if self.device == "cuda":
            flow_up = flow_up.cpu()
        flow = flow_up.detach().numpy()[0]
        flow = flow[:, : self.shape[0], : self.shape[1]]
        return flow


class GeneOpticalFlow:
    """Computation of optical flow for gene dataset.

    Parameters
    ----------
    gene_data
        Gene Dataset. It contains a ``volume`` of reference shape
        with all known slices located at the right place and a ``metadata``
        dictionary containing information about the axis of the dataset and the
        section numbers.
    reference_volume
        Reference volume used to compute optical flow. It needs to be of same
        shape as ``gene_data.volume``.
    model
        Model computing flow between two images.
    """

    def __init__(
        self, gene_data: GeneDataset, reference_volume: np.ndarray, model: OpticalFlow
    ):
        self.gene_data = gene_data
        self.reference_volume = reference_volume
        self.model = model

        self.axis = self.gene_data.axis

        if self.axis == "coronal":
            self.gene_volume = self.gene_data.volume.copy()
            self.reference_volume = reference_volume
        elif self.axis == "sagittal":
            self.gene_volume = np.moveaxis(self.gene_data.volume, 2, 0)
            self.reference_volume = np.moveaxis(reference_volume, 2, 0)
        else:
            raise ValueError(f"Unknown axis: {self.axis}")

        # Use to save all flow predictions
        self.pairwise_ref_flows = {}

    def predict_ref_flow(self, idx_from: int, idx_to: int) -> np.ndarray:
        """Compute optical flow between two given slices of the reference volume.

        Parameters
        ----------
        idx_from
            First section to consider
        idx_to
            Second section to consider.

        Returns
        -------
        flow : np.ndarray
            Predicted flow between the two given sections of the reference
            volume.

        Raises
        ------
        ValueError
            If one of the ``idx_from`` and ``idx_to`` is out of the boundaries
            of the reference space.
        """
        n_slices = len(self.gene_volume)
        if not (0 <= idx_from < n_slices and 0 <= idx_to < n_slices):
            raise ValueError(
                f"Slices ({idx_from} and {idx_to})"
                f"have to be between 0 and {n_slices}"
            )

        if (idx_from, idx_to) in self.pairwise_ref_flows:
            return self.pairwise_ref_flows[(idx_from, idx_to)]

        preimg1, preimg2 = self.model.preprocess_images(
            self.reference_volume[idx_from], self.reference_volume[idx_to]
        )
        flow = self.model.predict_flow(preimg1, preimg2)
        self.pairwise_ref_flows[(idx_from, idx_to)] = flow
        return flow

    def predict_slice(self, slice_number: int) -> np.ndarray:
        """Predict one gene slice.

        Parameters
        ----------
        slice_number
            Slice section to predict.

        Returns
        -------
        np.ndarray
            Predicted gene slice. Array of shape ``(dim1, dim2, 3)``
            being ``(528, 320)`` for sagittal dataset and
            ``(320, 456)`` for coronal dataset.
        """
        closest = self.gene_data.get_closest_known(slice_number)
        flow = self.predict_ref_flow(slice_number, closest)

        closest_slice = self.gene_volume[closest]
        predicted_slice = self.model.warp_image(flow, closest_slice)

        return predicted_slice

    def predict_volume(self) -> np.ndarray:
        """Predict entire volume with known gene slices.

        This function might be slow.

        Returns
        -------
        np.ndarray
            Entire gene volume. Array of shape of the volume ``GeneDataset``.
        """
        volume_shape = self.gene_volume.shape
        volume = np.zeros(volume_shape)

        # Populate the volume
        for slice_number in range(volume.shape[0]):
            # If the slice is known, just copy the gene.
            if slice_number in self.gene_data.known_slices:
                volume[slice_number] = self.gene_volume[slice_number]
            # If the slice is unknown, predict it
            else:
                volume[slice_number] = self.predict_slice(slice_number)

        if self.gene_data.axis == "sagittal":
            volume = np.moveaxis(volume, 0, 2)

        return volume
