"""Tests for optical models."""
import argparse
from unittest import mock

import numpy as np
import pytest
import torch

from atlinter.optical_flow import MaskFlowNet, OpticalFlow, RAFTNet


class DummyModel(OpticalFlow):
    def predict_flow(self, img1, img2):
        raise NotImplementedError


def test_optical_flow_model():
    """Test Optical Flow."""
    shape = (10, 10)
    img1 = np.random.rand(*shape)
    img2 = np.random.rand(*shape)
    flow = np.random.rand(2, *shape)

    model = DummyModel()

    pre_img1, pre_img2 = model.preprocess_images(img1, img2)
    warped_img2 = model.warp_image(flow, img2)
    assert pre_img1 is img1
    assert pre_img2 is img2
    assert warped_img2.shape == shape


def test_raftnet_model(monkeypatch):
    """Test RAFT model."""
    shape = (10, 10)
    img1 = np.random.rand(*shape)
    img2 = np.random.rand(*shape)

    fake_load = mock.Mock(return_value={})
    fake_torch_parallel = mock.Mock()
    fake_load_state = mock.Mock()
    fake_torch_parallel.load_state_dict.return_value = fake_load_state
    monkeypatch.setattr(
        "atlinter.optical_flow.torch.nn.DataParallel", fake_torch_parallel
    )
    monkeypatch.setattr("atlinter.optical_flow.torch.load", fake_load)

    raft = RAFTNet(path="test", device="cpu")
    raft.model = mock.Mock(
        spec=torch.nn.Module,
        return_value=[torch.rand(1, 2, 10, 10), torch.rand(1, 2, 10, 10)],
    )

    # ValueError when shape of images are different.
    with pytest.raises(ValueError):
        raft.preprocess_images(img1, np.random.rand(20, 20))

    args = raft.initialize_namespace()
    assert isinstance(args, argparse.Namespace)

    preimg1, preimg2 = raft.preprocess_images(img1, img2)
    assert isinstance(preimg1, np.ndarray) and isinstance(preimg2, np.ndarray)
    expected_shape = (1, 3, 10, 10)
    assert (preimg1.shape == expected_shape) and (preimg2.shape == expected_shape)

    flow = raft.predict_flow(preimg1, preimg2)
    assert isinstance(flow, np.ndarray)
    assert flow.shape == (2, 10, 10)

    new_img1 = raft.warp_image(flow, img2)
    assert isinstance(new_img1, np.ndarray)
    assert new_img1.shape == shape


def test_maskflownet_model(monkeypatch):
    """Test RAFT model."""
    shape = (10, 10)
    img1 = np.random.rand(*shape)
    img2 = np.random.rand(*shape)

    fake_network = mock.Mock()
    fake_network.get_pipeline.return_value = "pipe"
    monkeypatch.setattr("atlinter.optical_flow.MaskFlowNet.network", fake_network)
    fake_predict_new_data = mock.Mock()
    fake_predict_new_data.load_checkpoint.return_value = "model"
    fake_predict_new_data.predict_image_pair_flow.return_value = (
        np.random.rand(10, 10, 2),
        img1,
        img1,
    )
    monkeypatch.setattr(
        "atlinter.optical_flow.MaskFlowNet.predict_new_data", fake_predict_new_data
    )

    maskflow = MaskFlowNet(checkpoint_path="fake")
    with pytest.raises(ValueError):
        maskflow.preprocess_images(img1, np.random.rand(20, 20))

    preimg1, preimg2 = maskflow.preprocess_images(img1, img2)
    assert isinstance(preimg1, np.ndarray) and isinstance(preimg2, np.ndarray)
    expected_shape = (10, 10, 3)
    assert (preimg1.shape == expected_shape) and (preimg2.shape == expected_shape)

    flow = maskflow.predict_flow(preimg1, preimg2)
    assert isinstance(flow, np.ndarray)
    assert flow.shape == (2, 10, 10)

    new_img1 = maskflow.warp_image(flow, img2)
    assert isinstance(new_img1, np.ndarray)
    assert new_img1.shape == shape
