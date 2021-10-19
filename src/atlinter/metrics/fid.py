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
"""Implementation of the Fréchet Inception Distance (FID) metric."""
import numpy as np
import torch
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3


class FID:
    """Class for computing the Fréchet Inception Distance (FID) metric score.

    It is implemented as a class in order to hold the inception model instance
    in its state.

    Parameters
    ----------
    resize_input : bool (optional)
        Whether or not to resize the input images to the image size (299, 299)
        on which the inception model was trained. Since the model is fully
        convolutional, the score also works without resizing. In literature
        and when working with GANs people tend to set this value to True,
        however, for internal evaluation this is not necessary.
    device : str or torch.device
        The device on which to run the inception model.
    """

    def __init__(self, resize_input=False, device=None):
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = InceptionV3(resize_input=resize_input).to(device)
        self.model = self.model.eval()

    def _get_mu_sigma(self, images):
        """Compute the inception mu and sigma for a batch of images.

        Parameters
        ----------
        images : np.ndarray
            A batch of images with shape (n_images, width, height).

        Returns
        -------
        mu : np.ndarray
            The array of mean activations with shape (2048,).
        sigma : np.ndarray
            The covariance matrix of activations with shape (2048, 2048).
        """
        # forward pass
        batch = torch.tensor(images).unsqueeze(1).repeat((1, 3, 1, 1))
        batch = batch.to(self.device, torch.float32)
        (activations,) = self.model(batch)
        activations = activations.detach().cpu().numpy().squeeze(3).squeeze(2)

        # compute statistics
        mu = activations.mean(axis=0)
        sigma = np.cov(activations, rowvar=False)

        return mu, sigma

    def score(self, images_1, images_2):
        """Compute the FID score.

        The input batches should have the shape (n_images, width, height).

        Parameters
        ----------
        images_1 : np.ndarray
            First batch of images.
        images_2 : np.ndarray
            Section batch of images.

        Returns
        -------
        score : float
            The FID score.
        """
        mu_1, sigma_1 = self._get_mu_sigma(images_1)
        mu_2, sigma_2 = self._get_mu_sigma(images_2)
        score = calculate_frechet_distance(mu_1, sigma_1, mu_2, sigma_2)

        return score
