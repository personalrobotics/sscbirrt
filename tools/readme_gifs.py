# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Regenerate the README's demo GIFs.

    uv run python tools/readme_gifs.py                    # all of them
    uv run python tools/readme_gifs.py transport door     # only these scenarios

pick: one GIF per seed (docs/images/pick_<can>_seed<N>.gif). transport: the upright carry (transport.gif). door: the
reach and the opening (door.gif).

Each GIF is rendered without the text overlay (the README captions each one), at 1024x576 scaled to 400 wide,
12 fps, through a two-pass palette with the ffmpeg that imageio-ffmpeg ships. The motion plays in real time: each
scenario's timed trajectory under the demo limits (#214). Needs ``sscbirrt[demo]``.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from sscbirrt.demo import headless_gl_default

headless_gl_default()

import imageio_ffmpeg  # noqa: E402

from sscbirrt.demo.render import render_video  # noqa: E402
from sscbirrt.demo.scenarios import door, pick, transport  # noqa: E402

# Seeds chosen from a sweep of 0-59 (sscbirrt 3.4 defaults, timed under #214) for variety: every can, different
# grasps and routes, all planned fast. Seed 54 is the sweep's only blue can.
SEEDS = [5, 9, 54, 20, 58, 37]
OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
FILTER = (
    "fps=12,scale=400:-1:flags=lanczos,split[a][b];"
    "[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
)


def write_gif(outcome, clips, target: Path) -> None:
    """Render ``clips`` without the overlay and convert to a palette GIF at ``target``."""
    with tempfile.TemporaryDirectory() as tmp:
        mp4 = Path(tmp) / "clip.mp4"
        render_video(
            outcome.model,
            outcome.data,
            clips,
            mp4,
            camera=outcome.camera,
            width=1024,
            height=576,
            overlay=False,
        )
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(mp4), "-vf", FILTER, "-loop", "0",
             str(target)],
            check=True,
        )  # fmt: skip
    print(f"{target.relative_to(OUT.parent.parent)}  {target.stat().st_size / 1e6:.2f} MB")


def main(argv: list[str]) -> int:
    chosen = set(argv) or {"pick", "transport", "door"}
    if "pick" in chosen:
        for seed in SEEDS:
            outcome = pick.run(seed)
            if not outcome.ok:
                print(f"pick seed {seed}: {outcome.report[-1]}", file=sys.stderr)
                return 1
            can = outcome.report[1].split("the ")[1].split(" can")[0]
            write_gif(outcome, outcome.clips, OUT / f"pick_{can}_seed{seed}.gif")
    if "transport" in chosen:
        outcome = transport.run(0)
        if not outcome.ok:
            print(f"transport: {outcome.report[-1]}", file=sys.stderr)
            return 1
        write_gif(outcome, outcome.clips, OUT / "transport.gif")
    if "door" in chosen:
        outcome = door.run(0)
        if not outcome.ok:
            print(f"door: {outcome.report[-1]}", file=sys.stderr)
            return 1
        write_gif(outcome, outcome.clips, OUT / "door.gif")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
