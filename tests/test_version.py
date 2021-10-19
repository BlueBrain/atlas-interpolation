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
import atlinter


def test_version_exists():
    # Version exists
    assert hasattr(atlinter, "__version__")
    assert isinstance(atlinter.__version__, str)
    parts = atlinter.__version__.split(".")

    # Version has correct format
    # allow for an optional ".devXX" part for local testing
    assert len(parts) in {3, 4}
    assert parts[0].isdecimal()  # major
    assert parts[1].isdecimal()  # minor
    assert parts[2].isdecimal()  # patch
