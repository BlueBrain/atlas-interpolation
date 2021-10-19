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
import numpy as np
import pytest

from atlinter.data import GeneDataset


class TestGeneData:
    @pytest.mark.parametrize("axis", ["coronal", "sagittal"])
    @pytest.mark.parametrize("rgb", [True, False])
    def test_gene_data(self, axis, rgb):
        """Test GeneData class."""
        if rgb:
            volume_shape = (10, 20, 30, 3)
        else:
            volume_shape = (10, 20, 30)
        end = volume_shape[0] if axis == "coronal" else volume_shape[2]
        gene_shape = (20, 30) if axis == "coronal" else (10, 20)
        if rgb:
            known_gene = np.random.rand(2, *gene_shape, 3)
        else:
            known_gene = np.random.rand(2, *gene_shape)
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
        if gene_data.axis == "sagittal" and rgb:
            volume = np.transpose(gene_data.volume, (2, 0, 1, 3))
        elif gene_data.axis == "sagittal" and not rgb:
            volume = np.transpose(gene_data.volume, (2, 0, 1))
        else:
            volume = gene_data.volume
        for i in range(end):
            if i in known_slices:
                assert volume[i].sum() > 0
            else:
                assert volume[i].sum() == 0

    @pytest.mark.parametrize("rgb", [True, False])
    def test_get_closest(self, rgb):
        """Test get closest known from GeneData."""
        if rgb:
            image_shape = (20, 30, 3)
        else:
            image_shape = (20, 30)

        known_gene = np.random.rand(10, *image_shape)
        known_slices = np.arange(1, 11) * 10
        # known_slices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        gene_data = GeneDataset(
            known_gene, known_slices, "coronal", volume_shape=(110, *image_shape)
        )
        assert gene_data.get_closest_known(10) == 10
        assert gene_data.get_closest_known(110) == 100
        assert gene_data.get_closest_known(78) == 80
        assert gene_data.get_closest_known(55) == 50

    @pytest.mark.parametrize("rgb", [True, False])
    def test_get_surroundings(self, rgb):
        """Test get surrounding gene slices."""
        if rgb:
            image_shape = (20, 30, 3)
        else:
            image_shape = (20, 30)
        known_gene = np.random.rand(10, *image_shape)
        known_slices = np.arange(1, 11) * 10
        # known_slices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        gene_data = GeneDataset(
            known_gene, known_slices, "coronal", volume_shape=(110, *image_shape)
        )
        assert gene_data.get_surrounding_slices(2) == (None, 10)
        assert gene_data.get_surrounding_slices(10) == (10, 20)
        assert gene_data.get_surrounding_slices(22) == (20, 30)
        assert gene_data.get_surrounding_slices(200) == (100, None)
