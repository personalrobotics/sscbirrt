# Contributing

This repository is part of the [robot-code workspace](https://github.com/personalrobotics/robot-code).
All projects in the workspace share a common layout (`src/` + `tests/`), a single
package manager (`uv`), and a common CI setup.

## Development setup

```bash
git clone https://github.com/personalrobotics/robot-code
cd robot-code
./setup.sh
```

This clones every sibling repo (including this one) into a single uv workspace
and runs `uv sync`. You can then work in any sibling's directory.

## Running tests and linters

From inside this repo:

```bash
uv run pytest tests/ -v
uv run ruff check .
uv run ruff format --check .
```

From the workspace root, you can also run the cross-repo integration suite:

```bash
uv run pytest tests/integration -v
```

## Pull requests

- Branch from `main`. Open a PR with a clear summary and test plan.
- Per-repo CI (ruff + pytest) must pass.
- Cross-repo integration CI in robot-code runs automatically when this repo's
  `main` advances, and on scheduled nightly runs.
- Review is by [@siddhss5](https://github.com/siddhss5) (enforced via CODEOWNERS).

## Releasing

Releases are tag-driven (`.github/workflows/release.yml`). Each run builds every wheel and the sdist once, gathers
them with `SHA256SUMS` and a manifest, verifies those files in clean environments (`tools/release_check.py`), and
publishes the same bytes, re-checking their hashes at every step. Nothing is rebuilt after verification.

**The tagged commit's version is the tag's version, exactly.** `vX.Y.ZrcN` is cut from a commit whose
`pyproject.toml` version is `X.Y.ZrcN`; `vX.Y.Z` from one whose version is `X.Y.Z`. `project(VERSION X.Y.Z)` in
`cpp/CMakeLists.txt` is always the numeric release, since CMake cannot hold an rc suffix; the wheel's native core
reports the full package version. The workflow's first job refuses any disagreement before anything is built or
uploaded. Check a tag locally with `python3 tools/release_check.py version --tag vX.Y.ZrcN`.

1. **Release candidate.** Merge a PR that sets `version = "X.Y.Zrc1"` in `pyproject.toml`, `project(... VERSION X.Y.Z)`
   in `cpp/CMakeLists.txt`, and the `## [X.Y.Z]` CHANGELOG section. Tag the merge commit `vX.Y.Zrc1` and push the
   tag. The run builds, verifies, uploads to TestPyPI (no approval: `testpypi-release` only admits `v*` tags), downloads the
   files back, confirms their hashes, and reruns the clean wheel checks on the downloaded files. Its
   `release-report` artifact records the result. Nothing goes to PyPI.
2. **Final.** Merge a PR that sets `version = "X.Y.Z"` and dates the CHANGELOG section. Tag the merge commit
   `vX.Y.Z` and push the tag. The run builds the final files once, verifies them, uploads them to TestPyPI and
   confirms them there, uploads the same files to PyPI (the `release` environment admits only `v*` tags and waits for a maintainer's
   approval in the run's page) and confirms them there,
   installs from PyPI and runs the quick start, and creates the GitHub release from the CHANGELOG section with the
   distributions, `SHA256SUMS`, `manifest.json` and `release-report.json` attached.

**Never move or delete a tag while a release run for it exists** (running, waiting, or able to be re-run). If a tag
must change, cut the next version (the next rc, or the next patch) instead. The TestPyPI, PyPI and GitHub-release
jobs each stop unless the tag still points to the run's commit, so a moved tag fails the run rather than publishing
one commit's files under another; it does not make moving a tag safe.

A failure anywhere before the PyPI upload leaves PyPI untouched. A transient failure (an index slow to serve the
files, a runner fault) is retried with "Re-run failed jobs" on the same run, which reuses the gathered files. A
failure in the files themselves needs a new build, and so the next rc or patch version: TestPyPI refuses different
files under a version's existing names, and the tag stays where it is. `workflow_dispatch` builds, gathers, verifies and writes the report on any branch without publishing,
which is how to rehearse a change to the pipeline.
To check gathered files by hand, download the run's `release-dist` artifact and run
`python3 tools/release_check.py dist release-dist/dist --version X.Y.Z --sums release-dist/SHA256SUMS --python 3.12`.

## Package manager: uv only

We use [uv](https://docs.astral.sh/uv/) exclusively. **Do not use pip.**
The workspace layout in robot-code relies on uv's workspace resolution.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License (see [LICENSE](LICENSE)).
