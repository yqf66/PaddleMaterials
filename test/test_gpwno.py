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
        "num_local_gcn_layers": 1,
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


def test_legacy_local_branch_configuration_and_weights():
    model = build_small_model(model_sharing=False)
    restored = build_small_model(
        model_sharing=False,
        num_local_gcn_layers=None,
        num_infgcn_layer=1,
        input_infgcn=False,
    )
    assert restored.num_local_gcn_layers == 1
    assert not restored.input_local_density
    legacy_state = {
        name.replace("local_gcns.", "infgcns.", 1): tensor.clone()
        for name, tensor in model.state_dict().items()
    }
    assert any(name.startswith("infgcns.") for name in legacy_state)
    missing, unexpected = restored.set_state_dict(legacy_state)
    assert not missing
    assert not unexpected
    for name, tensor in model.state_dict().items():
        np.testing.assert_array_equal(
            tensor.numpy(), restored.state_dict()[name].numpy()
        )


@pytest.mark.parametrize(
    "options",
    [
        {"num_local_gcn_layers": 1, "num_infgcn_layer": 2},
        {"input_local_density": False, "input_infgcn": True},
    ],
)
def test_conflicting_legacy_local_branch_options(options):
    with pytest.raises(ValueError, match="Conflicting local"):
        build_small_model(**options)


@pytest.mark.parametrize("model_sharing", [False, True])
@pytest.mark.parametrize("periodic", [False, True])
def test_local_density_input_matches_independent_graphs(model_sharing, periodic):
    model = build_small_model(
        input_local_density=True,
        model_sharing=model_sharing,
        pbc=periodic,
        use_max_cell=not periodic,
        equivariant_frame=False,
    )
    model.eval()
    atom_types = paddle.to_tensor([0, 1, 2, 0], dtype="int64")
    atoms = paddle.to_tensor(
        [[0.2, 0.3, 0.4], [1.0, 0.3, 0.4], [0.3, 0.5, 0.4], [0.3, 1.3, 0.4]],
        dtype="float32",
    )
    grid = paddle.to_tensor(
        [[[0.5, 0.5, 0.5], [1.2, 0.8, 0.6]], [[0.6, 0.5, 0.8], [0.8, 1.1, 0.7]]],
        dtype="float32",
    )
    batch = paddle.to_tensor([0, 0, 1, 1], dtype="int64")
    cells = paddle.eye(3).unsqueeze(0).tile([2, 1, 1]) * 3
    info = {"cell": cells} if periodic else None
    with paddle.no_grad():
        prediction, _ = model._forward_density(atom_types, atoms, grid, batch, info)
        separate = []
        for index in range(2):
            selected = batch == index
            value, _ = model._forward_density(
                atom_types[selected],
                atoms[selected],
                grid[index : index + 1],
                paddle.zeros([2], dtype="int64"),
                {"cell": cells[index : index + 1]} if periodic else None,
            )
            separate.append(value)
    assert bool(paddle.isfinite(prediction).all())
    np.testing.assert_allclose(
        prediction.numpy(), paddle.concat(separate).numpy(), rtol=1e-5, atol=1e-5
    )
