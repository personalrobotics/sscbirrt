# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Release checks on the exact files a release publishes (#260).

The release workflow (``.github/workflows/release.yml``) builds every wheel and the sdist once, gathers them, and
carries those same files through verification, TestPyPI, PyPI and the GitHub release. This script is every step of
that path that is not an upload:

    release_check.py version --tag v3.13.0rc1
        The tag, pyproject's version and CMake's project(VERSION) agree: the tag's version equals pyproject's
        exactly (rc suffix included), and pyproject's numeric release part equals CMake's. Writes ``version``
        and ``kind`` (final, rc, or untagged) to $GITHUB_OUTPUT when it is set.

    release_check.py gather DIST --version 3.13.0rc1 --out DIR
        Every file in DIST is an sscbirrt wheel or the one sdist of that version, by filename and by the version
        recorded inside it. Writes DIR/SHA256SUMS (``sha256sum -c`` format) and DIR/manifest.json.

    release_check.py dist DIST --version V [--sums SHA256SUMS] [--install wheel sdist] [--report R.json]
        Re-checks the hashes, then for each install kind creates a clean virtual environment outside the source
        tree, installs that file with the demo extra, and runs the named checks in CHECKS from a work directory
        with ``python -I``. The README, tools and tests the checks run come from the gathered sdist.

    release_check.py index {testpypi,pypi} --version V --sums SHA256SUMS [--verify] [--quick-start]
        Downloads every file the index serves for that version and confirms the set of names and every hash
        against SHA256SUMS. --verify then runs the dist checks on the downloaded files (dependencies come from
        PyPI only); --quick-start installs ``sscbirrt[demo]==V`` from PyPI itself and runs the README.

    release_check.py report --manifest manifest.json --reports DIR --needs JSON --out release-report.json
        Merges the manifest, every check report, and the workflow's job results into the release report.

Every check report is JSON: the mode, the version, and one record per check with its name, install kind, status
(passed, failed or skipped), duration and output tail. The exit status is non-zero when any check fails.

Needs ``uv`` on PATH. Uses the standard library only, so it runs before anything is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "sscbirrt"
# A release tag is vX.Y.Z or vX.Y.ZrcN, in PEP 440's canonical spelling, so the tag, the package version and every
# filename agree byte for byte.
RELEASE_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?")
INDEXES = {"testpypi": "https://test.pypi.org", "pypi": "https://pypi.org"}


class CheckFailed(Exception):
    """A check's expected failure: the message is its diagnostic."""


# ---------------------------------------------------------------------------------------------------------------------
# version


def _cmake_version(root: Path) -> str:
    text = (root / "cpp" / "CMakeLists.txt").read_text()
    m = re.search(r"^project\(\s*sscbirrt\s+VERSION\s+([0-9.]+)", text, re.M)
    if not m:
        raise CheckFailed("cpp/CMakeLists.txt has no project(sscbirrt VERSION ...)")
    return m.group(1)


def check_version(root: Path, tag: str | None) -> tuple[str, str]:
    """Return (version, kind) or raise CheckFailed when the tag, pyproject and CMake disagree."""
    import tomllib  # Python 3.11+; the release jobs run 3.12

    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    m = RELEASE_VERSION.fullmatch(version)
    if not m:
        raise CheckFailed(f"pyproject version {version!r} is not X.Y.Z or X.Y.ZrcN")
    release = ".".join(m.group(i) for i in (1, 2, 3))
    cmake = _cmake_version(root)
    if cmake != release:
        raise CheckFailed(f"pyproject version {version} has release part {release}, but CMake's project is {cmake}")
    if tag is None:
        return version, "untagged"
    if not tag.startswith("v") or not RELEASE_VERSION.fullmatch(tag[1:]):
        raise CheckFailed(f"tag {tag!r} is not vX.Y.Z or vX.Y.ZrcN")
    if tag[1:] != version:
        raise CheckFailed(
            f"tag {tag} is version {tag[1:]}, but pyproject's version is {version}; they must agree exactly"
        )
    return version, "rc" if m.group(4) is not None else "final"


# ---------------------------------------------------------------------------------------------------------------------
# gather and hashes


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _metadata_version(path: Path) -> str:
    """The Version field recorded inside a wheel's METADATA or an sdist's PKG-INFO."""
    if path.name.endswith(".whl"):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if re.fullmatch(r"[^/]+\.dist-info/METADATA", n)]
            if len(names) != 1:
                raise CheckFailed(f"{path.name} has {len(names)} dist-info METADATA files")
            text = z.read(names[0]).decode()
    else:
        with tarfile.open(path) as t:
            member = t.getmember(path.name.removesuffix(".tar.gz") + "/PKG-INFO")
            text = t.extractfile(member).read().decode()
    m = re.search(r"^Version: (\S+)$", text, re.M)
    if not m:
        raise CheckFailed(f"{path.name} records no Version")
    return m.group(1)


def classify(dist: Path, version: str) -> dict[str, list[Path]]:
    """Split DIST into wheels and the sdist, refusing any file that is not this package at this version."""
    wheels, sdists, wrong = [], [], []
    for path in sorted(dist.iterdir()):
        if path.is_file() and re.fullmatch(rf"{PACKAGE}-{re.escape(version)}-[^/]+\.whl", path.name):
            wheels.append(path)
        elif path.is_file() and path.name == f"{PACKAGE}-{version}.tar.gz":
            sdists.append(path)
        else:
            wrong.append(path.name)
    if wrong:
        raise CheckFailed(f"not {PACKAGE} {version} distributions: {', '.join(wrong)}")
    if not wheels or len(sdists) != 1:
        raise CheckFailed(f"expected wheels and one sdist; found {len(wheels)} wheels and {len(sdists)} sdists")
    for path in wheels + sdists:
        inside = _metadata_version(path)
        if inside != version:
            raise CheckFailed(f"{path.name} records version {inside}, not {version}")
    return {"wheel": wheels, "sdist": sdists}


def read_sums(sums: Path) -> dict[str, str]:
    out = {}
    for line in sums.read_text().splitlines():
        m = re.fullmatch(r"([0-9a-f]{64})  ([^/\s]+)", line)
        if not m or m.group(2) in out:
            raise CheckFailed(f"{sums.name}: malformed or repeated line {line!r}")
        out[m.group(2)] = m.group(1)
    if not out:
        raise CheckFailed(f"{sums.name} is empty")
    return out


def recheck(dist: Path, sums: Path) -> dict[str, str]:
    """DIST holds exactly the files SHA256SUMS lists, each with its listed hash."""
    expected = read_sums(sums)
    present = {p.name for p in dist.iterdir()}
    if present != set(expected):
        raise CheckFailed(
            f"files differ from {sums.name}: missing {sorted(set(expected) - present)}, "
            f"extra {sorted(present - set(expected))}"
        )
    bad = [n for n, h in expected.items() if sha256(dist / n) != h]
    if bad:
        raise CheckFailed(f"hash mismatch: {', '.join(bad)}")
    return expected


def _write_json(path: Path, data: object) -> None:
    """Atomic replace, so a failed write never leaves a truncated report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
    os.replace(tmp, path)


def gather(dist: Path, version: str, out: Path, tag: str | None, commit: str | None, run_url: str | None) -> dict:
    kinds = classify(dist, version)
    files = [
        {"name": p.name, "kind": kind, "size": p.stat().st_size, "sha256": sha256(p)}
        for kind in ("sdist", "wheel")
        for p in kinds[kind]
    ]
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "SHA256SUMS.tmp"
    tmp.write_text("".join(f"{f['sha256']}  {f['name']}\n" for f in sorted(files, key=lambda f: f["name"])))
    os.replace(tmp, out / "SHA256SUMS")
    manifest = {
        "package": PACKAGE,
        "version": version,
        "tag": tag,
        "commit": commit,
        "run": run_url,
        "sha256sums_sha256": sha256(out / "SHA256SUMS"),
        "files": files,
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------------------------------------------------
# clean-environment checks


@dataclass
class Env:
    """One clean virtual environment with sscbirrt installed, and where its checks run."""

    kind: str  # wheel, sdist, or index
    python: Path
    work: Path
    source: Path  # the extracted sdist: README.md, tools/, tests/
    version: str
    env: dict[str, str] = field(default_factory=dict)

    def run(self, *args: str | Path, timeout: float = 3600) -> str:
        proc = subprocess.run(
            [str(a) for a in args],
            cwd=self.work,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise CheckFailed(f"exit {proc.returncode}: {' '.join(map(str, args))}\n{_tail(proc.stdout)}")
        return proc.stdout

    def py(self, *args: str | Path) -> str:
        return self.run(self.python, "-I", *args)

    @property
    def bin(self) -> Path:
        return self.python.parent


def _tail(text: str, lines: int = 40) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


VERSION_PROBE = """
import importlib.metadata as md, pathlib, sys
import sscbirrt, sscbirrt._native as n
from sscbirrt.backends import native_mujoco
expected, prefix = sys.argv[1], pathlib.Path(sys.prefix).resolve()
where = pathlib.Path(sscbirrt.__file__).resolve()
assert where.is_relative_to(prefix), f"sscbirrt imported from {where}, outside the clean environment {prefix}"
assert md.version("sscbirrt") == expected, (md.version("sscbirrt"), expected)
assert n.__version__ == expected, ("native core", n.__version__, expected)
assert n.has_ssik(), n.ssik_unavailable_reason()
assert native_mujoco.available(), native_mujoco.unavailable_reason()
print("package", md.version("sscbirrt"), "native", n.__version__, "from", where)
"""

# #247 and #206 through public names only: one scene, strict -> allowed -> strict for a held can on a table.
PUBLIC_ALLOWANCES = """
import numpy as np, mujoco, sscbirrt.backends.native_mujoco as nm
from sscbirrt.backends.native_mujoco import Attachment, NativeCollisionChecker, NativeScene, Snapshot
xml = '''<mujoco><worldbody>
  <body name="table" pos="0.555 0 0.5"><geom type="box" size="0.03 0.08 0.08"/></body>
  <body name="arm" pos="0 0 0.5"><joint name="j0" type="hinge" axis="0 0 1"/>
    <geom type="capsule" size="0.02" fromto="0 0 0 0.38 0 0"/>
    <body name="arm/gripper/base" pos="0.4 0 0"><geom type="box" size="0.02 0.03 0.02"/></body></body>
  <body name="can" pos="2 0 0.5"><freejoint/><geom type="cylinder" size="0.03 0.05"/></body>
</worldbody></mujoco>'''
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
scene = NativeScene.from_model(m, ["j0"])
T = np.eye(4); T[0, 3] = 0.1
def snap(allowed):
    held = Attachment("can", "arm/gripper/base", T, allowed_bodies=allowed)
    return Snapshot.from_state(scene, d.qpos, attachments=[held])
strict, allowed = snap(None), snap(["arm/gripper/base", "table"])
again = Snapshot.capture(scene, d, [Attachment("can", "arm/gripper/base", T)])
v = [NativeCollisionChecker(scene, s).is_valid(np.zeros(1)) for s in (strict, allowed, again)]
assert v == [False, True, False], v
assert strict.sha256 == again.sha256 != allowed.sha256
assert all(s.scene is scene for s in (strict, allowed, again))
private = [n for n in ("Attachment", "Snapshot", "NativeScene", "NativeCollisionChecker")
           if "_native_mujoco" in (getattr(nm, n).__doc__ or "") or "_load" in (getattr(nm, n).__doc__ or "")]
assert not private, private
print("allowances: strict -> allowed -> strict on one scene", v)
"""


def check_version_reported(e: Env) -> str:
    return e.py("-c", VERSION_PROBE, e.version)


def check_demos(e: Env) -> str:
    return e.run(e.bin / "sscbirrt-demo", "--no-video", "--out", e.work / "demo")


def check_reference_artifact(e: Env) -> str:
    return e.py(e.source / "tools" / "reference_artifact.py", "--backend", "native", "--check")


def check_validation_artifact(e: Env) -> str:
    return e.py(e.source / "tools" / "validation_artifact.py", "--check")


def check_readme(e: Env) -> str:
    return e.py("-m", "pytest", "-q", "-p", "no:cacheprovider", e.source / "tests" / "test_readme.py")


def check_public_api(e: Env) -> str:
    """Public names alone give the documented answers. Each release's public-behaviour checks append here."""
    return e.py("-c", PUBLIC_ALLOWANCES)


# The checks run, in order, against each clean install. A release that adds public behaviour appends its check.
CHECKS: list[tuple[str, Callable[[Env], str]]] = [
    ("version", check_version_reported),
    ("demos", check_demos),
    ("reference_artifact", check_reference_artifact),
    ("validation_artifact", check_validation_artifact),
    ("readme", check_readme),
    ("public_api", check_public_api),
]
QUICK_START = ("version", "readme")


def _uv(*args: str | Path, cwd: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise CheckFailed("uv is not on PATH")
    proc = subprocess.run(
        [uv, "--no-config", *map(str, args)], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if proc.returncode != 0:
        raise CheckFailed(f"uv {' '.join(map(str, args))}\n{_tail(proc.stdout)}")


def _clean_env(menagerie: str | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT")}
    if menagerie:
        env["MUJOCO_MENAGERIE_PATH"] = menagerie
    return env


def _platform_wheel(wheels: list[Path], python: Path) -> Path:
    """The one wheel built for this interpreter and platform."""
    tag = subprocess.run(
        [python, "-I", "-c", "import sys; print(f'cp{sys.version_info[0]}{sys.version_info[1]}')"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    plat = {"linux": "manylinux", "darwin": "macosx"}.get(sys.platform, sys.platform)
    arch = platform.machine().lower().replace("aarch64", "arm64")
    match = [w for w in wheels if f"-{tag}-{tag}-" in w.name and plat in w.name and arch in w.name]
    if len(match) != 1:
        raise CheckFailed(f"{len(match)} wheels for {tag} {plat} {arch} among {[w.name for w in wheels]}")
    return match[0]


def _extract_sdist(sdist: Path, into: Path) -> Path:
    with tarfile.open(sdist) as t:
        t.extractall(into, filter="data")
    return into / sdist.name.removesuffix(".tar.gz")


def _record(results: list[dict], name: str, kind: str, fn: Callable[[], str]) -> bool:
    start = time.monotonic()
    try:
        out = fn()
        status, detail = "passed", _tail(out, 12)
    except CheckFailed as exc:
        status, detail = "failed", str(exc)
    except subprocess.TimeoutExpired as exc:
        status, detail = "failed", f"timed out after {exc.timeout} s: {exc.cmd}"
    results.append(
        {
            "name": name,
            "install": kind,
            "status": status,
            "seconds": round(time.monotonic() - start, 1),
            "detail": detail,
        }
    )
    print(f"[{status}] {kind}: {name} ({results[-1]['seconds']} s)", flush=True)
    if status == "failed":
        print(detail, flush=True)
    return status == "passed"


def run_checks(
    kind: str,
    requirement: str | Callable[[Path], str],
    source: Path,
    version: str,
    python_request: str,
    work_root: Path,
    names: tuple[str, ...] | None,
    install_args: tuple[str, ...] = (),
    attempts: int = 1,
) -> list[dict]:
    """Install ``requirement`` (or the one it names for the new interpreter) into a fresh environment and run the
    checks; a failed install skips the rest."""
    results: list[dict] = []
    work = work_root / kind
    work.mkdir(parents=True)
    venv = work_root / f"{kind}-venv"
    python = venv / "bin" / "python"
    env = Env(kind, python, work, source, version, _clean_env())

    def install() -> str:
        _uv("venv", "--quiet", "--python", python_request, venv, cwd=work)
        target = requirement(python) if callable(requirement) else requirement
        for attempt in range(attempts):  # an index can list a new release before its simple page serves it
            try:
                _uv(
                    "pip",
                    "install",
                    "--quiet",
                    "--no-cache",
                    "--python",
                    python,
                    *install_args,
                    target,
                    "pytest",
                    cwd=work,
                )
                break
            except CheckFailed:
                if attempt == attempts - 1:
                    raise
                time.sleep(60)
        menagerie = env.py("-c", "import sscbirrt_assets; print(sscbirrt_assets.menagerie_path())").strip()
        env.env = _clean_env(menagerie)
        installed = env.py("-c", "import importlib.metadata as m; print(m.version('sscbirrt'))").strip()
        return f"installed sscbirrt {installed} from {Path(target).name}"

    chosen = [(n, f) for n, f in CHECKS if names is None or n in names]
    if not _record(results, "install", kind, install):
        results += [
            {"name": n, "install": kind, "status": "skipped", "seconds": 0, "detail": "install failed"}
            for n, _ in chosen
        ]
        return results
    for name, fn in chosen:
        _record(results, name, kind, lambda fn=fn: fn(env))
    return results


def verify_dist(dist: Path, version: str, installs: list[str], python_request: str, work_root: Path) -> list[dict]:
    kinds = classify(dist, version)
    source = _extract_sdist(kinds["sdist"][0], work_root / "source")
    results: list[dict] = []
    for kind in installs:
        if kind == "wheel":
            requirement = lambda python: f"{_platform_wheel(kinds['wheel'], python).resolve()}[demo]"  # noqa: E731
        else:
            requirement = f"{kinds['sdist'][0].resolve()}[demo]"
        results += run_checks(kind, requirement, source, version, python_request, work_root, None)
    return results


# ---------------------------------------------------------------------------------------------------------------------
# index


def _fetch_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def confirm_index(index: str, version: str, sums: Path, download: Path, wait: float) -> dict:
    """Download every file the index serves for VERSION and compare names and hashes with SHA256SUMS."""
    expected = read_sums(sums)
    url = f"{INDEXES[index]}/pypi/{PACKAGE}/{version}/json"
    deadline = time.monotonic() + wait
    while True:
        data = _fetch_json(url)
        served = {f["filename"]: f for f in (data or {}).get("urls", [])}
        if set(served) == set(expected) or time.monotonic() > deadline:
            break
        print(f"{index} serves {len(served)} of {len(expected)} files; waiting", flush=True)
        time.sleep(30)
    files = []
    download.mkdir(parents=True, exist_ok=True)
    for name in sorted(set(served) | set(expected)):
        record = {"name": name, "expected": expected.get(name), "index_reported": None, "downloaded": None}
        if name in served and name in expected:  # only names SHA256SUMS lists are written to disk
            record["index_reported"] = served[name]["digests"]["sha256"]
            target = download / name
            with urllib.request.urlopen(served[name]["url"], timeout=300) as r, target.open("wb") as f:
                shutil.copyfileobj(r, f)
            record["downloaded"] = sha256(target)
        record["match"] = record["downloaded"] is not None and (
            record["expected"] == record["index_reported"] == record["downloaded"]
        )
        files.append(record)
    return {"index": index, "url": url, "files": files, "all_match": bool(files) and all(f["match"] for f in files)}


# ---------------------------------------------------------------------------------------------------------------------
# report


def merge_report(manifest: Path, reports: Path, needs: str, out: Path) -> dict:
    m = json.loads(manifest.read_text()) if manifest.is_file() else None
    checks, indexes = [], {}
    for path in sorted(reports.rglob("*.json")) if reports.is_dir() else []:
        r = json.loads(path.read_text())
        checks += [dict(c, source=r.get("label", path.stem)) for c in r.get("checks", [])]
        if r.get("index"):
            indexes[r["index"]["index"]] = r["index"]
    jobs = {name: job.get("result") for name, job in json.loads(needs or "{}").items()}
    report = {
        "package": PACKAGE,
        "tag": (m or {}).get("tag"),
        "commit": (m or {}).get("commit"),
        "version": (m or {}).get("version"),
        "run": (m or {}).get("run"),
        "artifacts": (m or {}).get("files", []),
        "testpypi": indexes.get("testpypi"),
        "pypi": indexes.get("pypi"),
        "checks": checks,
        "jobs": jobs,
        "all_checks_passed": bool(checks) and all(c["status"] == "passed" for c in checks),
    }
    _write_json(out, report)
    return report


# ---------------------------------------------------------------------------------------------------------------------


def _finish(report_path: Path | None, report: dict) -> int:
    failed = [c for c in report.get("checks", []) if c["status"] != "passed"]
    if report.get("index") and not report["index"]["all_match"]:
        failed.append({"name": "index hashes"})
    report["passed"] = not failed
    if report_path:
        _write_json(report_path, report)
    print(json.dumps({"passed": report["passed"], "failed": [c["name"] for c in failed]}))
    return 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("version", help="the tag, pyproject and CMake agree")
    p.add_argument("--tag", help="the release tag (omit for an untagged build)")
    p.add_argument("--root", type=Path, default=ROOT)

    p = sub.add_parser("gather", help="classify the built files and write SHA256SUMS and manifest.json")
    p.add_argument("dist", type=Path)
    p.add_argument("--version", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tag")
    p.add_argument("--commit")
    p.add_argument("--run-url")

    for name in ("dist", "index"):
        p = sub.add_parser(name)
        if name == "dist":
            p.add_argument("dist", type=Path, help="the directory of gathered distribution files")
        else:
            p.add_argument("index", choices=sorted(INDEXES))
            p.add_argument("--download", type=Path, help="where to keep the downloaded files (default: a temp dir)")
            p.add_argument("--wait", type=float, default=900, help="seconds to wait for the index to serve all files")
            p.add_argument("--verify", action="store_true", help="run the dist checks on the downloaded files")
            p.add_argument(
                "--quick-start",
                action="store_true",
                help="install sscbirrt[demo]==VERSION from PyPI and run the README (pypi only)",
            )
        p.add_argument("--version", required=True)
        p.add_argument("--sums", type=Path, required=name == "index", help="SHA256SUMS to re-check first")
        p.add_argument("--install", nargs="+", choices=("wheel", "sdist"), default=["wheel", "sdist"])
        p.add_argument("--python", default="3.12", help="the interpreter version for the clean environments")
        p.add_argument("--work", type=Path, help="work directory, kept (default: a temp dir, removed)")
        p.add_argument("--label", help="this report's name in the release report")
        p.add_argument("--report", type=Path)

    p = sub.add_parser("report", help="merge the manifest, check reports and job results")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--reports", type=Path, required=True)
    p.add_argument("--needs", default=os.environ.get("NEEDS_JSON", "{}"))
    p.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.mode == "version":
            version, kind = check_version(args.root.resolve(), args.tag)
            print(f"version {version} ({kind})")
            if os.environ.get("GITHUB_OUTPUT"):
                with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                    f.write(f"version={version}\nkind={kind}\n")
            return 0
        if args.mode == "gather":
            manifest = gather(args.dist, args.version, args.out, args.tag, args.commit, args.run_url)
            for f in manifest["files"]:
                print(f"{f['sha256']}  {f['name']}")
            if os.environ.get("GITHUB_OUTPUT"):
                with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                    f.write(f"sums_sha256={manifest['sha256sums_sha256']}\n")
            return 0
        if args.mode == "report":
            report = merge_report(args.manifest, args.reports, args.needs, args.out)
            print(json.dumps({k: report[k] for k in ("tag", "version", "all_checks_passed", "jobs")}))
            return 0
    except CheckFailed as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    work_root = (args.work or Path(tempfile.mkdtemp(prefix="sscbirrt-release-check-"))).resolve()
    if work_root.is_relative_to(ROOT):
        print(f"error: the work directory {work_root} is inside the source tree {ROOT}", file=sys.stderr)
        return 1
    report: dict = {
        "mode": args.mode,
        "label": args.label or args.mode,
        "version": args.version,
        "python": args.python,
        "platform": f"{sys.platform}-{platform.machine()}",
        "checks": [],
    }
    try:
        if args.mode == "dist":
            dist = args.dist.resolve()
            if args.sums:
                recheck(dist, args.sums)
                report["sums"] = read_sums(args.sums)
            report["checks"] = verify_dist(dist, args.version, args.install, args.python, work_root)
        else:
            download = (args.download or work_root / "download").resolve()
            report["index"] = confirm_index(args.index, args.version, args.sums, download, args.wait)
            print(json.dumps({f["name"]: f["match"] for f in report["index"]["files"]}, indent=1))
            if report["index"]["all_match"] and args.verify:
                recheck(download, args.sums)
                report["checks"] += verify_dist(download, args.version, args.install, args.python, work_root)
            if report["index"]["all_match"] and args.quick_start:
                if args.index != "pypi":
                    raise CheckFailed("--quick-start installs from PyPI; it needs the pypi index")
                source = _extract_sdist(download / f"{PACKAGE}-{args.version}.tar.gz", work_root / "index-source")
                report["checks"] += run_checks(
                    "index",
                    f"{PACKAGE}[demo]=={args.version}",
                    source,
                    args.version,
                    args.python,
                    work_root,
                    QUICK_START,
                    ("--index-url", "https://pypi.org/simple"),
                    attempts=10,
                )
    except CheckFailed as exc:
        report["checks"].append(
            {"name": args.mode, "install": None, "status": "failed", "seconds": 0, "detail": str(exc)}
        )
        print(f"error: {exc}", file=sys.stderr)
    finally:
        if args.work is None:
            shutil.rmtree(work_root, ignore_errors=True)
    return _finish(args.report, report)


if __name__ == "__main__":
    sys.exit(main())
