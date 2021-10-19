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
"""Align ISH gene to Nissl slices."""
import argparse
import json
import pathlib
import sys

import numpy as np
from atlannot.ants import register, transform
from skimage.color import rgb2gray


def parse_args():
    """Parse command line arguments.

    Returns
    -------
    args : argparse.Namespace
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("gene_name")
    args = parser.parse_args()

    return args


def dataset_registration(nissl, gene, section_numbers, axis):
    """Align ISH data to a given nissl slice thanks to ANTsPY.

    Parameters
    ----------
    nissl : np.ndarray
        Corresponding nissl section for the given gene.
    gene : np.ndarray
        Dataset of gene expression images to align to Nissl.
    section_numbers : list
        List of section numbers of the corresponding genes
    axis : {'coronal', 'sagittal'}
        Axis of the gene expressions

    Returns
    -------
    aligned_gene : np.ndarray
        Aligned gene expressions.
    """
    aligned_gene = []
    for single_gene, section_number in zip(gene, section_numbers):
        if axis == "coronal":
            try:
                single_nissl = nissl[int(section_number), :, :]
            except IndexError:
                aligned_gene.append(single_gene)
                continue
        elif axis == "sagittal":
            single_nissl = nissl[:, :, int(section_number)]
        else:
            raise ValueError(
                f"Axis {axis} is not supported"
                "Only coronal and sagittal are supported!"
            )

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
    rgb = False
    tmp_gene = single_gene.copy()

    if len(single_gene.shape) == 3:
        rgb = True
        tmp_gene = 1 - rgb2gray(tmp_gene).astype(np.float32)

    if single_nissl.shape != tmp_gene.shape:
        raise ValueError(
            f"Nissl ({single_nissl.shape}) and gene "
            f"({tmp_gene.shape}) should have the same shape"
        )

    nii_data = register(single_nissl, tmp_gene)

    if rgb:
        aligned_single_gene = np.zeros_like(single_gene)
        aligned_single_gene[:, :, 0] = transform(
            single_gene[:, :, 0].astype(np.float32),
            nii_data,
            defaultvalue=single_gene[0, 0, 0],
        )
        aligned_single_gene[:, :, 1] = transform(
            single_gene[:, :, 1].astype(np.float32),
            nii_data,
            defaultvalue=single_gene[0, 0, 1],
        )
        aligned_single_gene[:, :, 2] = transform(
            single_gene[:, :, 2].astype(np.float32),
            nii_data,
            defaultvalue=single_gene[0, 0, 2],
        )
    else:
        aligned_single_gene = transform(single_gene, nii_data)

    return aligned_single_gene


def main():
    """Align gene expression to Nissl."""
    # Imports
    import nrrd

    args = parse_args()
    gene_name = args.gene_name

    cwd = pathlib.Path(__file__).parent.resolve()
    nissl_path = cwd / "ara_nissl_25.nrrd"
    if not nissl_path.exists():
        raise ValueError("Nissl volume is missing! Please dvc pull download-nissl")

    nissl, _ = nrrd.read(nissl_path)
    nissl = nissl / nissl.max()

    for axis in ["sagittal", "coronal"]:

        file_dir = cwd / axis / gene_name
        if not file_dir.exists():
            continue
        experiment_ids = [
            path.stem for path in file_dir.iterdir() if path.suffix == ".npy"
        ]

        saving_dir = cwd / "aligned" / axis / gene_name
        if not saving_dir.exists():
            saving_dir.mkdir(parents=True)

        for experiment in experiment_ids:
            # Loading
            gene = np.load(file_dir / f"{experiment}.npy")
            with open(file_dir / f"{experiment}.json") as f:
                metadata = json.load(f)
            section_numbers = metadata["section_numbers"]

            # Registration
            aligned_gene = dataset_registration(nissl, gene, section_numbers, axis)

            # Saving
            np.save(saving_dir / f"{experiment}.npy", aligned_gene)
            with open(saving_dir / f"{experiment}.json", "w") as f:
                json.dump(metadata, f)


if __name__ == "__main__":
    sys.exit(main())
