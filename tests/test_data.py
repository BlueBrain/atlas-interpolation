import numpy as np
import pytest

from atlinter.data import GeneDataset


class TestGeneData:
    @pytest.mark.parametrize("axis", ["coronal", "sagittal"])
    def test_gene_data(self, axis):
        """Test GeneData class."""
        volume_shape = (10, 20, 30, 3)
        end = volume_shape[0] if axis == "coronal" else volume_shape[2]
        gene_shape = (20, 30) if axis == "coronal" else (10, 20)
        known_gene = np.random.rand(2, *gene_shape, 3)
        known_slices = [5, 8]  # random

        gene_data = GeneDataset(
            known_gene,
            section_numbers=known_slices,
            axis=axis,
            volume_shape=volume_shape,
        )

        assert isinstance(gene_data.axis, str)
        assert gene_data.axis == axis
        assert gene_data.known_slices == sorted(known_slices)
        assert isinstance(gene_data.volume, np.ndarray)
        assert gene_data.volume.shape == volume_shape
        assert isinstance(gene_data.unknown_slices, list)
        assert len(gene_data.unknown_slices) == end - len(known_slices)

        # Check volume is well populated with known slices
        if gene_data.axis == "sagittal":
            volume = np.transpose(gene_data.volume, (2, 0, 1, 3))
        else:
            volume = gene_data.volume
        for i in range(end):
            if i in known_slices:
                assert volume[i].sum() > 0
            else:
                assert volume[i].sum() == 0

    def test_get_closest(self):
        """Test get closest known from GeneData."""
        known_gene = np.random.rand(10, 20, 30, 3)
        known_slices = np.arange(1, 11) * 10
        # known_slices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        gene_data = GeneDataset(
            known_gene, known_slices, "coronal", volume_shape=(110, 20, 30, 3)
        )
        assert gene_data.get_closest_known(10) == 10
        assert gene_data.get_closest_known(110) == 100
        assert gene_data.get_closest_known(78) == 80
        assert gene_data.get_closest_known(55) == 50

    def test_get_surroundings(self):
        """Test get surrounding gene slices."""
        known_gene = np.random.rand(10, 20, 30, 3)
        known_slices = np.arange(1, 11) * 10
        # known_slices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        gene_data = GeneDataset(
            known_gene, known_slices, "coronal", volume_shape=(110, 20, 30, 3)
        )
        assert gene_data.get_surrounding_slices(2) == (None, 10)
        assert gene_data.get_surrounding_slices(10) == (10, 20)
        assert gene_data.get_surrounding_slices(22) == (20, 30)
        assert gene_data.get_surrounding_slices(200) == (100, None)
