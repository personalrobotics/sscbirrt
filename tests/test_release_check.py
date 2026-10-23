# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The release gate refuses before any upload (#260): a tag whose version is not the package's, and a gathered file
that is not this package at this version, stop the workflow at its first job and at gather. The installed-artifact
public clearance check (#259) holds for this build."""

import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("tomllib")  # the release jobs run Python 3.12

_spec = importlib.util.spec_from_file_location(
    "release_check", Path(__file__).parent.parent / "tools" / "release_check.py"
)
release_check = sys.modules.setdefault("release_check", importlib.util.module_from_spec(_spec))
_spec.loader.exec_module(release_check)


def _tree(tmp_path: Path, version: str, cmake: str) -> Path:
    (tmp_path / "cpp").mkdir()
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "sscbirrt"\nversion = "{version}"\n')
    (tmp_path / "cpp" / "CMakeLists.txt").write_text(f"project(sscbirrt VERSION {cmake} LANGUAGES CXX)\n")
    return tmp_path


@pytest.mark.parametrize(
    "version, cmake, tag, kind",
    [
        ("3.13.0rc1", "3.13.0", "v3.13.0rc1", "rc"),
        ("3.13.0", "3.13.0", "v3.13.0", "final"),
        ("3.13.0rc1", "3.13.0", None, "untagged"),
    ],
)
def test_agreeing_versions_pass(tmp_path, version, cmake, tag, kind):
    assert release_check.check_version(_tree(tmp_path, version, cmake), tag) == (version, kind)


@pytest.mark.parametrize(
    "version, cmake, tag",
    [
        ("3.13.0", "3.13.0", "v3.13.0rc1"),  # an rc tag on the final version: the old rehearsal, now refused
        ("3.13.0rc1", "3.13.0", "v3.13.0"),  # a final tag on an rc commit
        ("3.13.0rc1", "3.13.0", "v3.13.0rc2"),
        ("3.13.0", "3.12.0", "v3.13.0"),  # CMake left behind
        ("3.13.0", "3.13.0", "v3.13"),
        ("3.13.0", "3.13.0", "3.13.0"),
        ("3.13.0", "3.13.0", "v3.13.0-rc1"),
        ("3.13.0", "3.13.0", "v3.13.0.post1"),
        ("3.13.0", "3.13.0", "v3.13.0\n"),
        ("3.13.0.dev1", "3.13.0", None),
    ],
)
def test_disagreeing_versions_are_refused(tmp_path, version, cmake, tag):
    with pytest.raises(release_check.CheckFailed):
        release_check.check_version(_tree(tmp_path, version, cmake), tag)


def _wheel(path: Path, recorded: str) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            f"sscbirrt-{recorded}.dist-info/METADATA", f"Metadata-Version: 2.4\nName: sscbirrt\nVersion: {recorded}\n"
        )


def _sdist(path: Path, version: str) -> None:
    data = f"Metadata-Version: 2.4\nName: sscbirrt\nVersion: {version}\n".encode()
    info = tarfile.TarInfo(f"sscbirrt-{version}/PKG-INFO")
    info.size = len(data)
    with tarfile.open(path, "w:gz") as t:
        t.addfile(info, io.BytesIO(data))


def test_gather_writes_sums_that_recheck_and_refuses_a_file_that_is_not_this_version(tmp_path):
    dist, out = tmp_path / "dist", tmp_path / "out"
    dist.mkdir()
    _sdist(dist / "sscbirrt-3.13.0rc1.tar.gz", "3.13.0rc1")
    _wheel(dist / "sscbirrt-3.13.0rc1-cp312-cp312-manylinux_2_28_x86_64.whl", "3.13.0rc1")
    manifest = release_check.gather(dist, "3.13.0rc1", out, "v3.13.0rc1", "abc", None)
    assert set(release_check.recheck(dist, out / "SHA256SUMS")) == {f["name"] for f in manifest["files"]}

    # A changed byte, an extra file, and a wheel whose recorded version differs from its filename are all refused.
    (dist / "sscbirrt-3.13.0rc1.tar.gz").write_bytes(b"tampered")
    with pytest.raises(release_check.CheckFailed, match="hash mismatch"):
        release_check.recheck(dist, out / "SHA256SUMS")
    _sdist(dist / "sscbirrt-3.13.0rc1.tar.gz", "3.13.0rc1")
    (dist / "notes.txt").write_text("x")
    with pytest.raises(release_check.CheckFailed, match="extra"):
        release_check.recheck(dist, out / "SHA256SUMS")
    with pytest.raises(release_check.CheckFailed, match="notes.txt"):
        release_check.gather(dist, "3.13.0rc1", out, None, None, None)
    (dist / "notes.txt").unlink()
    _wheel(dist / "sscbirrt-3.13.0rc1-cp312-cp312-manylinux_2_28_x86_64.whl", "3.13.0")
    with pytest.raises(release_check.CheckFailed, match="records version 3.13.0,"):
        release_check.gather(dist, "3.13.0rc1", out, None, None, None)


def test_public_clearance_check_passes_on_the_installed_build():
    """The release's public clearance check (#259) holds for this build, through public names only."""
    pytest.importorskip("mujoco")
    from sscbirrt.backends import native_mujoco

    if not native_mujoco.available():
        pytest.skip(native_mujoco.unavailable_reason())
    assert "_load" not in release_check.PUBLIC_CLEARANCE and "_native_mujoco" not in release_check.PUBLIC_CLEARANCE
    out = subprocess.run(
        [sys.executable, "-I", "-c", release_check.PUBLIC_CLEARANCE], capture_output=True, text=True, check=True
    ).stdout
    result = json.loads(out.strip().splitlines()[-1])
    assert result["answers_0_and_5mm"] == {"0.0049": [True, False], "0.0051": [True, True]}
    assert result["clearance_m"] == [0.0, 0.005]
    assert set(result["refusals"]) == {"override", "flex"}
    assert result["malformed"] == {"True": "TypeError", "nan": "ValueError", "-0.001": "ValueError"}
