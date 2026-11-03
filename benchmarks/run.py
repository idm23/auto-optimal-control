"""Run the benchmarks and compare with the literature.

    uv run python benchmarks/run.py                 # all
    uv run python benchmarks/run.py robot_arm earth_mars
    uv run python benchmarks/run.py --markdown benchmarks/RESULTS.md
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from problems import ALL  # noqa: E402


def table(results):
    lines = ["| Benchmark | Quantity | Indirect (this package) | Reference | Rel. diff | Source | Time (s) |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r.benchmark} | {r.quantity} | {r.value:.10g} | {r.reference:.10g} | "
                     f"{r.rel_error:.1e} {'ok' if r.ok else 'FAIL'} | {r.source} | {r.seconds:.1f} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="benchmarks to run (default: all)")
    parser.add_argument("--cache-dir", default=None, help="kernel build cache")
    parser.add_argument("--markdown", default=None, help="write the results table to this file")
    args = parser.parse_args(argv)
    names = args.names or list(ALL)
    results = []
    for name in names:
        t0 = time.perf_counter()
        rows = ALL[name](cache_dir=args.cache_dir)
        results += rows
        for r in rows:
            print(f"{r.benchmark:16s} {r.quantity:40s} {r.value:.10g}  ref {r.reference:.10g}  "
                  f"rel {r.rel_error:.1e} {'ok' if r.ok else 'FAIL'}  {r.details}")
        print(f"  ({time.perf_counter() - t0:.1f} s)")
    text = table(results)
    print("\n" + text)
    if args.markdown:
        Path(args.markdown).write_text(text + "\n")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
