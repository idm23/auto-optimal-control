"""Animate how a benchmark is solved.

    uv run --group viz python benchmarks/animate.py cart_pole
    uv run --group viz python benchmarks/animate.py cart_pole --out outputs/cart_pole.gif

Recording runs the solves (about a minute for the cart-pole). It is saved
next to the output as ``<name>.npz``, and ``--reuse`` renders from it again
without solving.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from visuals import (bryson_denham, cart_pole, earth_mars, speed_limit,  # noqa: E402
                     velocity_limit)

SUITES = {"cart_pole": cart_pole, "earth_mars": earth_mars, "bryson_denham": bryson_denham,
          "velocity_limit": velocity_limit, "speed_limit": speed_limit}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", choices=list(SUITES))
    parser.add_argument("--out", default=None,
                        help="output .mp4 (needs ffmpeg) or .gif; default outputs/<name>.mp4")
    parser.add_argument("--stills", action="store_true",
                        help="also save the last frame of each stage as PNG")
    parser.add_argument("--reuse", action="store_true", help="render the saved recording")
    parser.add_argument("--n-random", type=int, default=400, help="random starts (cart_pole)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cache-dir", default=None, help="kernel build cache")
    args = parser.parse_args(argv)
    suite = SUITES[args.name]
    out = Path(args.out or f"outputs/{args.name}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = out.with_suffix(".npz")
    if args.reuse:
        data = suite.load(rec)
    else:
        extra = {"n_random": args.n_random} if args.name == "cart_pole" else {}
        data = suite.record(cache_dir=args.cache_dir, **extra)
        suite.save(data, rec)
    suite.animate(data, out, fps=args.fps, stills_dir=out.parent if args.stills else None)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
