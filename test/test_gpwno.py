# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import pytest

from ppmat.models.gpwno.GPWNO import GPWNO


def build_small_model(**overrides):
    params = {
        "n_atom_type": 3,
        "num_radial": 2,
        "num_spherical": 1,
        "radial_embed_size": 4,
        "radial_hidden_size": 4,
        "num_fourier": 4,
        "num_fourier_time": 2,
        "num_gcn_layer": 1,
        "num_infgcn_layer": 1,
        "width": 2,
        "padding": 0,
        "max_cell_size": 4,
        "probe_cutoff": 3,
        "grid_cutoff": 3,
        "residual": False,
    }
    params.update(overrides)
    return GPWNO(**params)


@pytest.mark.parametrize(
    "model_sharing,rno_degree,expected_degree",
    [(True, None, 1), (True, 2, 1), (False, None, 1), (False, 2, 2)],
)
def test_resolved_rno_degree(model_sharing, rno_degree, expected_degree):
    model = build_small_model(model_sharing=model_sharing, num_spherical_RNO=rno_degree)
    assert model.irreps_sh_RNO.lmax == expected_degree
    assert model.irreps_feat_RNO.dim == 2 * (expected_degree + 1) ** 2


@pytest.mark.parametrize("using_ff", [False, True])
@pytest.mark.parametrize("activation", ["norm", "scalar"])
def test_centered_prediction_preserves_inputs_and_translation(using_ff, activation):
    paddle.seed(42)
    model = build_small_model(using_ff=using_ff, activation=activation)
    model.eval()
    atoms = paddle.to_tensor(
        [
            [0.25, 0.125, 0.0625],
            [1.125, 0.25, 0.5],
            [0.375, 1.75, 0.75],
            [0.5, 0.625, 2.25],
        ],
        dtype="float32",
    )
    grid = paddle.to_tensor(
        [[[0.5, 0.5, 0.5], [1.0, 0.75, 1.25], [0.25, 1.0, 1.5]]],
        dtype="float32",
    )
    atom_types = paddle.to_tensor([0, 1, 2, 0], dtype="int64")
    batch = paddle.zeros([4], dtype="int64")
    original_atoms, original_grid = atoms.numpy().copy(), grid.numpy().copy()

    with paddle.no_grad():
        density, _ = model._forward_density(atom_types, atoms, grid, batch, None)
        repeated, _ = model._forward_density(atom_types, atoms, grid, batch, None)
        translation = paddle.to_tensor([4.0, -2.0, 1.0])
        translated, _ = model._forward_density(
            atom_types, atoms + translation, grid + translation, batch, None
        )

    np.testing.assert_array_equal(atoms.numpy(), original_atoms)
    np.testing.assert_array_equal(grid.numpy(), original_grid)
    assert np.isfinite(density.numpy()).all()
    np.testing.assert_allclose(density.numpy(), repeated.numpy(), atol=1e-6)
    np.testing.assert_allclose(density.numpy(), translated.numpy(), atol=1e-5)
