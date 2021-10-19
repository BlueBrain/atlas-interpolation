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
"""Align special volumes to Nissl slices."""
import argparse
import pathlib
import sys

import numpy as np
from atlannot.ants import register, transform


def parse_args():
    """Parse command line arguments.

    Returns
    -------
    args : argparse.Namespace
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_id")
    args = parser.parse_args()

    return args


def dataset_registration(nissl, gene):
    """Align gene dataset to Nissl.

    Parameters
    ----------
    nissl : np.ndarray
        Corresponding nissl section for the given gene.
    gene : np.ndarray
        Dataset of gene expression images to align to Nissl.

    Returns
    -------
    aligned_gene : np.ndarray
        Aligned gene expressions.
    """
    aligned_gene = []
    for section_number, single_gene in enumerate(gene):
        if np.sum(single_gene) == 0:
            aligned_gene.append(single_gene)
            continue
        else:
            try:
                single_nissl = nissl[int(section_number), :, :]
            except IndexError:
                aligned_gene += [
                    single_gene,
                ]
                continue

        aligned_gene += [
            single_registration(single_nissl, single_gene),
        ]

    aligned_gene = np.array(aligned_gene)
    return aligned_gene


def single_registration(single_nissl, single_gene):
    """Align ISH data to a given nissl slice thanks to ANTsPY.

    Parameters
    ----------
    single_nissl : np.ndarray
        Corresponding nissl section for the given gene.
    single_gene : np.ndarray
        Gene expression to align to Nissl.

    Returns
    -------
    aligned_single_gene : np.ndarray
        Aligned gene expression.
    """
    tmp_gene = single_gene.copy()

    if single_nissl.shape != tmp_gene.shape:
        raise ValueError(
            f"Nissl ({single_nissl.shape}) and gene "
            f"({tmp_gene.shape}) should have the same shape"
        )

    if np.max(single_gene) > 1:
        tmp_gene = single_gene / single_gene.max()

    nii_data = register(single_nissl, tmp_gene.astype(np.float32))
    aligned_single_gene = transform(single_gene.astype(np.float32), nii_data)

    return aligned_single_gene


def main():
    """Align gene expression to Nissl."""
    # Imports
    import nrrd

    args = parse_args()
    dataset_id = args.dataset_id

    cwd = pathlib.Path(__file__).parent.resolve()
    nissl_path = cwd / "ara_nissl_25.nrrd"
    if not nissl_path.exists():
        raise ValueError("Nissl volume is missing! Please dvc pull download-nissl")

    nissl, _ = nrrd.read(nissl_path)
    nissl = nissl / nissl.max()

    file_dir = cwd / "special_volumes" / f"dataset_{dataset_id}"
    if not file_dir.exists():
        raise ValueError(f"The dataset {dataset_id} does not exist.")

    experiment_ids = [
        path.stem
        for path in file_dir.iterdir()
        if path.suffix == ".npy" and path.stem != "dfs"
    ]

    saving_dir = cwd / "aligned" / "special_volumes" / f"dataset_{dataset_id}"
    if not saving_dir.exists():
        saving_dir.mkdir(parents=True)

    for experiment in experiment_ids:
        # Loading
        gene = np.load(file_dir / f"{experiment}.npy")

        # Registration
        aligned_gene = dataset_registration(nissl, gene)

        # Saving
        np.save(saving_dir / f"{experiment}.npy", aligned_gene)


if __name__ == "__main__":
    sys.exit(main())
