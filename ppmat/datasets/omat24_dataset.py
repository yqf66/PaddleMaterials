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


class OMat24Dataset(ASELMDBDataset):
    """OMat24 rattled-500 structures in the original ASE-LMDB layout.

    Only train and val archives are provided for this subset. A local path
    points directly to an ASE-LMDB file or a directory containing such files.
    """

    def __init__(self, build_graph_cfg, split, path=None, **kwargs):
        if split not in {"train", "val"}:
            raise ValueError("OMat24 rattled-500 supports only train and val splits.")
        if path is not None and not osp.exists(path):
            raise FileNotFoundError(path)
        if path is None:
            url = (
                "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OMat24/"
                f"{split}/rattled-500.tar.gz"
            )
            cache_dir = osp.join(download.DATASETS_HOME, "OMat24", split)
            path = download.get_path_from_url(url, cache_dir)
        super().__init__(build_graph_cfg=build_graph_cfg, path=path, **kwargs)
