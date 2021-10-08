from unittest import mock

import numpy as np
import pytest
import torch

from atlinter.data import GeneDataset
from atlinter.pair_interpolation import (
    AntsPairInterpolationModel,
    CAINPairInterpolationModel,
    GeneInterpolate,
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


class FakeModel(PairInterpolationModel):
    def interpolate(self, img1, img2):
        img_middle = (img1 + img2) / 2
        return img_middle


def fake_gene_data(axis, shape):
    n_known_slices = 3
    gene_volume = np.ones((n_known_slices, *shape, 3))
    for i in range(n_known_slices):
        gene_volume[i] = (i + 1) * gene_volume[i]

    section_numbers = [10 + i * 8 for i in range(n_known_slices)]

    return GeneDataset(gene_volume, section_numbers, axis, (30, 20, 40, 3))


class TestGeneInterpolate:
    def test_get_n_repeat(self):
        n_rep = [0, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]
        for diff, n in enumerate(n_rep):
            assert GeneInterpolate.get_n_repeat(diff) == n

    @pytest.mark.parametrize(
        ("left", "right", "n_repeat", "expected_output"),
        [
            (10, 11, 0, []),
            (10, 20, 1, np.array([15])),
            (10, 12, 2, np.array([10.5, 11, 11.5])),
            (10, 18, 3, np.arange(11, 18)),
            (10, 15, 3, [10.625, 11.25, 11.875, 12.5, 13.125, 13.75, 14.375]),
        ],
    )
    def test_get_predicted_section_numbers(
        self, left, right, n_repeat, expected_output
    ):
        """Check that predicted section numbers are good"""
        assert np.all(
            GeneInterpolate.get_predicted_section_numbers(left, right, n_repeat)
            == expected_output
        )

    @pytest.mark.parametrize(
        "axis",
        [
            "coronal",
            "sagittal",
        ],
    )
    def test_predict_slice(self, axis):
        """Check that one can predict one slice."""
        if axis == "coronal":
            shape = (20, 40)
        else:
            shape = (30, 20)

        gene_data = fake_gene_data(axis, shape)
        gene_interpolate = GeneInterpolate(gene_data, FakeModel())
        prediction = gene_interpolate.predict_slice(5)
        assert isinstance(prediction, np.ndarray)
        assert np.all(prediction.shape == (*shape, 3))

        # Slice to predict is < first known gene slice, copy the first one
        assert np.all(gene_interpolate.predict_slice(5) == np.ones((*shape, 3)) * 1)
        # Slice to predict is > last known gene slice, copy the last one
        assert np.all(gene_interpolate.predict_slice(100) == np.ones((*shape, 3)) * 3)
        # Result of the first iteration
        assert np.all(gene_interpolate.predict_slice(14) == np.ones((*shape, 3)) * 1.5)
        # Result of the second iteration
        assert np.all(gene_interpolate.predict_slice(12) == np.ones((*shape, 3)) * 1.25)
        assert np.all(gene_interpolate.predict_slice(16) == np.ones((*shape, 3)) * 1.75)
        # Result of the third iteration
        assert np.all(
            gene_interpolate.predict_slice(11) == np.ones((*shape, 3)) * 1.125
        )
        assert np.all(
            gene_interpolate.predict_slice(17) == np.ones((*shape, 3)) * 1.875
        )

    @pytest.mark.parametrize(
        "axis",
        [
            "coronal",
            "sagittal",
        ],
    )
    def test_get_all_predictions(self, axis):
        """Check that one can predict entire volume."""
        if axis == "coronal":
            shape = (20, 40)
        else:
            shape = (30, 20)

        gene_data = fake_gene_data(axis, shape)
        gene_interpolate = GeneInterpolate(gene_data, FakeModel())
        (
            all_interpolated_images,
            all_predicted_section_numbers,
        ) = gene_interpolate.get_all_interpolation()
        assert isinstance(all_interpolated_images, np.ndarray)
        assert isinstance(all_predicted_section_numbers, np.ndarray)
        assert np.all(all_predicted_section_numbers.shape == (14,))
        # 14 = 2 intervals * 7 interpolated images per interval
        assert np.all(all_interpolated_images.shape == (14, *shape, 3))

    @pytest.mark.parametrize(
        "axis",
        [
            "coronal",
            "sagittal",
        ],
    )
    def test_predict_volume(self, axis):
        """Check that one can predict entire volume."""
        if axis == "coronal":
            shape = (20, 40)
        else:
            shape = (30, 20)

        gene_data = fake_gene_data(axis, shape)
        gene_interpolate = GeneInterpolate(gene_data, FakeModel())
        predicted_volume = gene_interpolate.predict_volume()
        assert isinstance(predicted_volume, np.ndarray)
        assert np.all(predicted_volume.shape == (30, 20, 40, 3))

        if axis == "sagittal":
            predicted_volume = np.transpose(predicted_volume, (2, 0, 1, 3))
        assert np.all(np.unique(predicted_volume[0:10]) == np.array([1]))
        # Check only 2 slices (could check [36:]), to avoid too long test
        assert np.all(predicted_volume[36:38] == np.array([3]))

    def test_predict_volume_wrong_axis(self):

        gene_data = GeneDataset(
            np.zeros([1, 10, 10, 3]), [], "fake_axis", (10, 10, 10, 3)
        )
        gene_interpolate = GeneInterpolate(gene_data, FakeModel())
        with pytest.raises(ValueError):
            _ = gene_interpolate.predict_volume()
