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
from unittest import mock

import numpy as np
import torch

from atlinter.metrics import FID
from atlinter.metrics.fid import InceptionV3


@mock.patch("atlinter.metrics.fid.InceptionV3", spec=InceptionV3)
def test_fid(inception_class):
    # Input data
    n_images = 10
    images_1 = np.random.rand(n_images, 5, 5)
    images_2 = np.random.rand(n_images, 5, 5)

    # Mock for the inception class instance and its forward pass
    activations = torch.rand((n_images, 10, 1, 1), dtype=torch.float32)
    inception_instance = inception_class.return_value
    inception_instance.to.return_value = inception_instance
    inception_instance.eval.return_value = inception_instance
    inception_instance.return_value = [activations]

    # Test device is set automatically
    fid = FID()
    assert fid.device is not None
    assert fid.device in {"cuda", "cpu"}

    # Test score computation
    fid = FID(device="cpu")
    score = fid.score(images_1, images_2)
    assert inception_instance.call_count == 2
    assert isinstance(score, float)

    # Test two identical stacks of images have FID=0
    score = fid.score(images_1, images_1)
    assert np.allclose(score, 0)
