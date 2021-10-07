"""Optical flow computation based on pair of images."""
import argparse
import logging
from abc import ABC, abstractmethod

import mxnet as mx
import numpy as np
import torch
from scipy import ndimage

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
            The optical flow of shape (*image.shape, 2).
        """

    @staticmethod
    def warp_image(flow, img2, order=1):
        """Warp image with the predicted flow.

        Parameters
        ----------
        flow : np.ndarray
            The predicted optical flow of shape (*image.shape, 2).
        img2 : np.ndarray
            The right image.
        order : int
            The interpolation order. 0 = nearest neighbour, 1 = linear, 2 = cubic, etc.

        Returns
        -------
        warped_image : np.ndarray
            The warped image.
        """
        if len(img2.shape) != 2:
            raise ValueError(f"img must have shape (ni, nj), but got {img2.shape}")
        ni, nj = img2.shape
        ij = np.mgrid[:ni, :nj]
        warped_image = ndimage.map_coordinates(img2, ij - flow, order=order)
        return warped_image


class MaskFlowNet(OpticalFlow):
    """MaskFlowNet model for optical flow computation.

    The typical use is
    >>> from atlinter.optical_flow import MaskFlowNet
    >>> checkpoint_path = "data/checkpoints/maskflownet.params"
    >>> net = MaskFlowNet(checkpoint_path)

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint of the model. See references for possibilities.
    gpu_device : str
        Device to load optical flow model.
    """

    from atlinter.vendor.MaskFlowNet import network, predict_new_data

    def __init__(self, checkpoint_path, gpu_device="0"):
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
    >>> path = "data/checkpoints/RAFT/models/raft-things.pth"
    >>> net = RAFTNet(path)

    Parameters
    ----------
    path : str
        Path to the RAFT model.
    device : {'cpu', 'cuda'}
        Device to load optical flow model.
    """

    from atlinter.vendor.RAFT.raft import RAFT

    def __init__(self, path, device="cuda"):
        self.device = device
        args = self.initialize_namespace()
        self.model = torch.nn.DataParallel(self.RAFT(args))
        self.model.load_state_dict(torch.load(path))
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

        if np.max(img1) <= 1:
            img1 = (img1 * 255).astype(np.uint8)
            img2 = (img2 * 255).astype(np.uint8)

        if len(img1.shape) == 2:
            img1 = np.repeat(img1[np.newaxis], 3, axis=0)
            img2 = np.repeat(img2[np.newaxis], 3, axis=0)

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
        return flow
