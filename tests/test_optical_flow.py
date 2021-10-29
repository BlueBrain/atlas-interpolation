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
"""Tests for optical models."""
import argparse
from unittest import mock

import numpy as np
import pytest
import torch

from atlinter.data import GeneDataset
from atlinter.optical_flow import GeneOpticalFlow, MaskFlowNet, OpticalFlow, RAFTNet


class DummyModel(OpticalFlow):
    def predict_flow(self, img1, img2):
        raise NotImplementedError


@pytest.mark.parametrize("rgb", [True, False])
def test_optical_flow_model(rgb):
    """Test Optical Flow."""
    if rgb:
        shape = (10, 10, 3)
    else:
        shape = (10, 10)

    img1 = np.random.rand(*shape)
    img2 = np.random.rand(*shape)
    flow = np.random.rand(2, 10, 10)

    model = DummyModel()

    pre_img1, pre_img2 = model.preprocess_images(img1, img2)
    warped_img2 = model.warp_image(flow, img2)
    assert pre_img1 is img1
    assert pre_img2 is img2
    assert warped_img2.shape == shape

    with pytest.raises(ValueError) as exc:
        img2 = np.ones((20, 20, 3))
        model.warp_image(flow, img2)
    assert "The flow shape and the image shape should be consistent." in str(exc.value)

    with pytest.raises(ValueError) as exc:
        img2 = np.ones((10, 10, 1, 1, 5))
        model.warp_image(flow, img2)
    assert "Invalid image shape:" in str(exc.value)


@pytest.mark.parametrize("rgb", [True, False])
def test_raftnet_model(monkeypatch, rgb):
    """Test RAFT model."""
    if rgb:
        shape = (10, 10, 3)
    else:
        shape = (10, 10)
    img1 = np.random.rand(*shape)
    img2 = np.random.rand(*shape)

    if rgb:
        img1 = (img1 * 255).astype(np.uint8)
        img2 = (img2 * 255).astype(np.uint8)

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
    expected_shape = (1, 3, 16, 16)
    assert (preimg1.shape == expected_shape) and (preimg2.shape == expected_shape)
    assert preimg1.dtype == np.uint8 and preimg2.dtype == np.uint8
    assert 1 < np.max(preimg1) <= 255 and 1 < np.max(preimg2) <= 255

    flow = raft.predict_flow(preimg1, preimg2)
    assert isinstance(flow, np.ndarray)
    assert flow.shape == (2, 10, 10)

    new_img1 = raft.warp_image(flow, img2)
    assert isinstance(new_img1, np.ndarray)
    assert new_img1.shape == shape


@pytest.mark.parametrize("rgb", [True, False])
def test_maskflownet_model(monkeypatch, rgb):
    """Test RAFT model."""
    if rgb:
        shape = (10, 10, 3)
    else:
        shape = (10, 10)
    img1 = np.random.rand(*shape)
    img2 = np.random.rand(*shape)

    if rgb:
        img1 = (img1 * 255).astype(np.uint8)
        img2 = (img2 * 255).astype(np.uint8)

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
    assert preimg1.dtype == np.uint8 and preimg2.dtype == np.uint8
    assert 1 < np.max(preimg1) <= 255 and 1 < np.max(preimg2) <= 255

    flow = maskflow.predict_flow(preimg1, preimg2)
    assert isinstance(flow, np.ndarray)
    assert flow.shape == (2, 10, 10)

    new_img1 = maskflow.warp_image(flow, img2)
    assert isinstance(new_img1, np.ndarray)
    assert new_img1.shape == shape


class FakeModel(OpticalFlow):
    def predict_flow(self, img1, img2):
        return np.random.randn(2, *img1.shape[:2])


def instantiate_gene_optical_flow(gene_shape, volume_shape, axis):

    gene_data = GeneDataset(
        np.ones((3, *gene_shape)),
        [3, 8, 12],  # random
        axis=axis,
        volume_shape=volume_shape,
    )
    fake_model = FakeModel()
    reference_space = np.random.randn(*volume_shape)
    return GeneOpticalFlow(gene_data, reference_space, fake_model)


class TestGeneOpticalFlow:
    @pytest.mark.parametrize("axis", ["coronal", "sagittal"])
    @pytest.mark.parametrize("rgb", [True, False])
    @pytest.mark.parametrize("volume_shape", [(20, 30, 40), (40, 10, 35)])
    def test_predict_ref_flow(self, axis, rgb, volume_shape):
        """Test get prediction."""
        if axis == "coronal":
            gene_shape = (volume_shape[1], volume_shape[2])
        else:
            gene_shape = (volume_shape[0], volume_shape[1])

        if rgb:
            volume_shape = (*volume_shape, 3)
            gene_shape = (*gene_shape, 3)

        gene_optical_flow = instantiate_gene_optical_flow(
            gene_shape, volume_shape, axis
        )
        assert gene_optical_flow.axis == axis
        assert gene_optical_flow.pairwise_ref_flows == {}

        prediction_1 = gene_optical_flow.predict_ref_flow(0, 3)
        assert isinstance(prediction_1, np.ndarray)
        assert prediction_1.shape == (2, *gene_shape[0:2])
        assert (0, 3) in gene_optical_flow.pairwise_ref_flows
        prediction_2 = gene_optical_flow.predict_ref_flow(0, 3)
        assert np.all(prediction_1 == prediction_2)

        with pytest.raises(ValueError):
            # Out of boundary of the reference space
            gene_optical_flow.predict_ref_flow(0, 100)

    @pytest.mark.parametrize("axis", ["coronal", "sagittal"])
    @pytest.mark.parametrize("rgb", [True, False])
    @pytest.mark.parametrize("volume_shape", [(20, 30, 40), (40, 10, 35)])
    def test_predict_slice(self, axis, rgb, volume_shape):
        """Test prediction of slice"""
        if axis == "coronal":
            gene_shape = (volume_shape[1], volume_shape[2])
        else:
            gene_shape = (volume_shape[0], volume_shape[1])

        if rgb:
            volume_shape = (*volume_shape, 3)
            gene_shape = (*gene_shape, 3)

        gene_optical_flow = instantiate_gene_optical_flow(
            gene_shape, volume_shape, axis
        )
        predicted_slice = gene_optical_flow.predict_slice(10)
        assert isinstance(predicted_slice, np.ndarray)
        assert predicted_slice.shape == gene_shape

    @pytest.mark.parametrize("axis", ["coronal", "sagittal"])
    @pytest.mark.parametrize("rgb", [True, False])
    @pytest.mark.parametrize("volume_shape", [(20, 30, 40), (40, 10, 35)])
    def test_predict_volume(self, axis, rgb, volume_shape):
        """Test prediction of the volume"""
        if axis == "coronal":
            gene_shape = (volume_shape[1], volume_shape[2])
        else:
            gene_shape = (volume_shape[0], volume_shape[1])

        if rgb:
            volume_shape = (*volume_shape, 3)
            gene_shape = (*gene_shape, 3)

        gene_optical_flow = instantiate_gene_optical_flow(
            gene_shape, volume_shape, axis
        )
        predicted_volume = gene_optical_flow.predict_volume()
        assert isinstance(predicted_volume, np.ndarray)
        assert predicted_volume.shape == volume_shape
        # Check that the known slices are kept in the volume
        if axis == "sagittal":
            predicted_volume = np.moveaxis(predicted_volume, 2, 0)
        assert np.all(predicted_volume[[3, 8, 12]] == np.ones((3, *gene_shape)))
