# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import pytest

from ppmat.datasets.ase_lmdb_dataset import ASELMDBDataset
from ppmat.datasets.ase_lmdb_dataset import OMat24Dataset
from ppmat.utils import download


def test_omat24_downloads_each_split_into_separate_cache(monkeypatch, tmp_path):
    calls = []
    initialized = []

    def fetch(url, root):
        calls.append((url, root))
        return str(tmp_path / root.split("/")[-1] / "rattled-500")

    monkeypatch.setattr(download, "DATASETS_HOME", str(tmp_path))
    monkeypatch.setattr(download, "get_path_from_url", fetch)
    monkeypatch.setattr(
        ASELMDBDataset, "__init__", lambda self, **kwargs: initialized.append(kwargs)
    )
    for split in ("train", "val"):
        OMat24Dataset(build_graph_cfg={}, split=split)
    assert calls[0][0].endswith("/OMat24/train/rattled-500.tar.gz")
    assert calls[1][0].endswith("/OMat24/val/rattled-500.tar.gz")
    assert calls[0][1] != calls[1][1]
    assert initialized[0]["path"] != initialized[1]["path"]


def test_omat24_rejects_unavailable_test_split():
    with pytest.raises(ValueError, match="train and val"):
        OMat24Dataset(build_graph_cfg={}, split="test")


def test_omat24_missing_local_path_is_not_silently_downloaded(tmp_path):
    with pytest.raises(FileNotFoundError):
        OMat24Dataset(build_graph_cfg={}, split="val", path=str(tmp_path / "missing"))
