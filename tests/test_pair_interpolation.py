from unittest import mock

import numpy as np
import pytest
import torch

from atlinter.pair_interpolation import (
    AntsPairInterpolationModel,
    CAINPairInterpolationModel,
    LinearPairInterpolationModel,
    PairInterpolate,
    PairInterpolationModel,
    RIFEPairInterpolationModel,
)


class DummyModel(PairInterpolationModel):
    def interpolate(self, img1, img2):
        raise NotImplementedError


def test_pair_interpolation_model():
    # Test input
    img1 = np.random.rand()
    img2 = np.random.rand()
    interpolated_images = np.random.rand()

    # Instantiate model
    model = DummyModel()

    # Tests
    ret1, ret2 = model.before_interpolation(img1, img2)
    ret = model.after_interpolation(interpolated_images)
    assert ret1 is img1
    assert ret2 is img2
    assert ret is interpolated_images


@pytest.mark.parametrize(
    ("n_repeat", "n_expected_interpolations"),
    (
        (1, 1),
        (2, 3),
        (3, 7),
        (10, 1023),
    ),
)
def test_pair_interpolate(n_repeat, n_expected_interpolations):
    # Test input
    shape = (10, 10)
    img1 = np.random.rand(*shape)
    img2 = np.random.rand(*shape)

    # Interpolation model mock
    model = mock.Mock(spec=PairInterpolationModel)
    dummy_model = DummyModel()
    model.before_interpolation = mock.Mock(wraps=dummy_model.before_interpolation)
    model.interpolate = mock.Mock(return_value=np.random.rand(*shape))
    model.after_interpolation = mock.Mock(wraps=dummy_model.after_interpolation)

    # Run interpolation
    interpolate = PairInterpolate().repeat(n_repeat)
    interpolated = interpolate(img1, img2, model)

    # Check results
    assert isinstance(interpolated, np.ndarray)
    assert len(interpolated.shape) == 3
    assert interpolated.shape[1:] == img1.shape
    assert len(interpolated) == n_expected_interpolations
    model.before_interpolation.assert_called_once()
    assert model.interpolate.call_count == n_expected_interpolations
    model.after_interpolation.assert_called_once()


def test_linear_pair_interpolation_model():
    # Test input
    shape = (10, 10)
    img1 = np.random.rand(*shape)
    img2 = np.random.rand(*shape)

    # Test instances
    model = LinearPairInterpolationModel()
    interpolate = PairInterpolate()

    # Tests
    interpolated = interpolate(img1, img2, model)
    assert interpolated.shape == (1, *shape)
    assert np.allclose(interpolated[0], np.mean([img1, img2], axis=0))


def test_ants_pair_interpolation_model():
    # Test input
    shape = (10, 10)
    img1 = np.random.rand(*shape)
    img2 = np.random.rand(*shape)
    img_mid = np.random.rand(*shape)

    # Test instances
    register_fn = mock.Mock(return_value=np.random.rand())
    transform_fn = mock.Mock(return_value=img_mid)
    model = AntsPairInterpolationModel(register_fn, transform_fn)
    interpolate = PairInterpolate()

    # Tests
    interpolated = interpolate(img1, img2, model)
    assert interpolated.shape == (1, *shape)
    register_fn.assert_called_once()
    transform_fn.assert_called_once()


def test_cain_pair_interpolation_model():
    # Test input
    shape = (10, 10)
    img1 = np.random.rand(*shape)
    img2 = np.random.rand(*shape)
    img_mid = torch.rand(1, 3, *shape)

    # Test instances
    cain_model = mock.Mock(return_value=(img_mid, None))
    model = CAINPairInterpolationModel(cain_model)
    interpolate = PairInterpolate()

    # Tests
    interpolated = interpolate(img1, img2, model)
    assert interpolated.shape == (1, *shape)
    cain_model.assert_called_once()


def test_rife_pair_interpolation_model():
    # Test input
    shape = (10, 10)
    img1 = np.random.rand(*shape)
    img2 = np.random.rand(*shape)
    img_mid = torch.rand(1, 3, *shape)

    # Test instances
    rife_model = mock.Mock()
    rife_model.inference = mock.Mock(return_value=img_mid)
    rife_device = None
    model = RIFEPairInterpolationModel(rife_model, rife_device)
    interpolate = PairInterpolate()

    # Tests
    interpolated = interpolate(img1, img2, model)
    assert interpolated.shape == (1, *shape)
    rife_model.inference.assert_called_once()
