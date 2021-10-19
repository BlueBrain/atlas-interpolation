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
"""Gene dataset module."""
from __future__ import annotations

import numpy as np

from atlinter.utils import find_closest


class GeneDataset:
    """Class representing gene dataset.

    Parameters
    ----------
    gene
        Array containing all gene slices known from a given dataset.
        Dimensions should correspond to (n_genes, dim1, dim2)
        for grayscale images and (n_genes, dim1, dim2, 3) for RGB.
    section_numbers
        List of the section numbers of the gene slices.
    axis
        Axis of the gene dataset.
    volume_shape
        Shape of the final volume.
        Dimensions should correspond to (coronal, transverse, sagittal)
        for grayscale images and (coronal, transverse, sagittal, 3) for RGB.
    """

    def __init__(
        self,
        gene: np.ndarray,
        section_numbers: list[int],
        axis: str,
        volume_shape: tuple[int, int, int]
        | tuple[int, int, int, int] = (528, 320, 456, 3),
    ):
        """Instantiate gene dataset."""
        if not isinstance(gene, np.ndarray) or gene.ndim not in {3, 4}:
            raise ValueError(
                "The gene has to be an array of \n"
                "* 3 dimensions (n_genes, dim1, dim2) for grayscale images. \n"
                "* 4 dimensions (n_genes, dim1, dim2, 3) for RBG images."
            )
        self.gene = gene
        self.known_slices = section_numbers
        self.axis = axis
        self.volume_shape = volume_shape

        # Create volume
        self.volume = np.zeros(volume_shape)
        end = volume_shape[0] if self.axis == "coronal" else volume_shape[2]
        slices = [s for s in self.known_slices if s < end]

        # Populate volume with known gene slices
        if self.axis == "coronal":
            self.volume[slices] = self.gene
        elif self.axis == "sagittal":
            self.volume[:, :, slices] = np.moveaxis(self.gene, 0, 2)

        # Sorting known slices
        self.known_slices = sorted(self.known_slices)
        # Compute the unknown slices
        all_slices = np.arange(end)
        self.unknown_slices = [int(s) for s in all_slices if s not in self.known_slices]

    def get_closest_known(self, slice_number: int) -> int:
        """Obtain closest gene slice known for a given slice number."""
        return find_closest(slice_number, self.known_slices)[1]

    def get_surrounding_slices(
        self, slice_number: int
    ) -> tuple[int | None, int | None]:
        """Get the two known surrounding gene slices for a given slice number.

        If index is smaller than all indices, the result will be (None, smallest indice)
        If index is bigger than all indices, the result will be (biggest indice, None)
        If index is equal to one of the indices, the result will be (index, next indice)

        Parameters
        ----------
        slice_number : int
            Index to look for surrounding gene slices

        Returns
        -------
        left : int
            Left value to the index among indices.
        right : int
            Right value to the index among indices.
        """
        # Special cases
        if slice_number < self.known_slices[0]:
            return None, self.known_slices[0]
        if slice_number >= self.known_slices[-1]:
            return self.known_slices[-1], None

        # Find the first index that is strictly bigger than the given one.
        # After the special cases the indices list is at least of length 2.
        right_pos = 0
        while slice_number >= self.known_slices[right_pos]:
            right_pos += 1

        return self.known_slices[right_pos - 1], self.known_slices[right_pos]
