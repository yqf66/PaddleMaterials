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

import os.path as osp

from ppmat.datasets.ase_lmdb_dataset import ASELMDBDataset
from ppmat.utils import download


class UMAOC20Dataset(ASELMDBDataset):
    """Prepared OC20 S2EF data used by the UMA configuration."""

    url = (
        "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/UMA/uma_datasets.zip"
    )
    dataset_subdir = osp.join("oc20", "uma_aselmdb")

    def __init__(self, build_graph_cfg, split, path=None, **kwargs):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}.")
        if path is None or not osp.exists(path):
            root = download.get_datasets_path_from_url(self.url)
            path = self._get_downloaded_split_path(root, split)
        super().__init__(build_graph_cfg=build_graph_cfg, path=path, **kwargs)

    def _get_downloaded_split_path(self, root_path, split):
        relative_path = osp.join(self.dataset_subdir, split)
        for root in (root_path, osp.join(root_path, "uma_datasets")):
            path = osp.join(root, relative_path)
            if osp.exists(path):
                return path
        raise FileNotFoundError(f"Cannot find OC20 {split} under {root_path}.")
