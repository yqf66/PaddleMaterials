import zipfile
from pathlib import Path

import pytest

from ppmat.utils.download import _uncompress_file_zip
from ppmat.utils.model_package import resolve_model_package_dir


@pytest.mark.parametrize("directory_entry", [False, True])
@pytest.mark.parametrize("weights_first", [False, True])
def test_uncompress_single_directory_preserves_archive_root(
    tmp_path, directory_entry, weights_first
):
    archive_path = tmp_path / "model.zip"
    members = [
        ("model/checkpoints/best.pdparams", b"weights"),
        ("model/model.yaml", b"Model: {}"),
    ]
    if not weights_first:
        members.reverse()
    with zipfile.ZipFile(archive_path, "w") as archive:
        if directory_entry:
            archive.writestr("model/", b"")
        for name, content in members:
            archive.writestr(name, content)

    extracted_path = _uncompress_file_zip(str(archive_path))

    assert Path(extracted_path) == tmp_path / "model"
    package_dir = Path(resolve_model_package_dir("model", extracted_path))
    assert package_dir == tmp_path / "model" / "model"
    assert (package_dir / "model.yaml").is_file()
    assert (package_dir / "checkpoints" / "best.pdparams").is_file()
