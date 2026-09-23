#!/usr/bin/env python3
"""Run the three CLI backends and generate benchmarks/1.png through 4.png."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = ROOT / "benchmarks"
BACKENDS = {
    "eigen": ROOT / "x64" / "EIGEN_Release" / "Watermarking-CLI.exe",
    "cuda": ROOT / "x64" / "CUDA_Release" / "Watermarking-CLI.exe",
    "opencl": ROOT / "x64" / "OPENCL_Release" / "Watermarking-CLI.exe",
}
BACKEND_LABELS = {"eigen": "CPU", "cuda": "CUDA", "opencl": "OpenCL"}
BACKEND_COLORS = {"eigen": "#071d2b", "cuda": "#00bd68", "opencl": "#0795ad"}
RESOLUTIONS = ("480p", "720p", "1080p", "4K")
OPERATIONS = ("embed", "detect")


def run_benchmarks(opencl_device_id: int | None, loops: int | None) -> None:
    BENCHMARK_DIR.mkdir(exist_ok=True)
    for backend, executable in BACKENDS.items():
        if not executable.is_file():
            raise FileNotFoundError(f"Missing {backend} Release CLI: {executable}")
        command = [str(executable), "--bench"]
        if loops is not None:
            command.extend(("--benchmark_loops", str(loops)))
        if backend == "opencl" and opencl_device_id is not None:
            command.extend(("--opencl_device_id", str(opencl_device_id)))
        print(f"\n=== Running {BACKEND_LABELS[backend]} benchmark ===", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


def read_results() -> dict[str, list[dict[str, str]]]:
    results: dict[str, list[dict[str, str]]] = {}
    required_columns = {"backend", "device", "p", "resolution", "operation", "fps", "loops"}
    for backend in BACKENDS:
        csv_path = BENCHMARK_DIR / f"{backend}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing benchmark data: {csv_path}. Run with --run first.")
        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            missing = required_columns.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{csv_path} is missing columns: {', '.join(sorted(missing))}")
            rows = list(reader)
        if len(rows) != 32:
            raise ValueError(f"Expected 32 rows in {csv_path}, found {len(rows)}")
        results[backend] = rows
    return results


def generate_figures(results: dict[str, list[dict[str, str]]]) -> None:
    try:
        import matplotlib
        import numpy as np
        from matplotlib.ticker import FuncFormatter
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Figure generation requires matplotlib and numpy: python -m pip install matplotlib numpy") from error

    backend_order = ("eigen", "cuda", "opencl")
    for figure_number, p in enumerate((3, 5, 7, 9), start=1):
        categories = [(resolution, operation) for resolution in RESOLUTIONS for operation in OPERATIONS]
        labels = [f"{resolution} ME {'make' if operation == 'embed' else 'Corr'}" for resolution, operation in categories]
        x_positions = np.arange(len(categories), dtype=float)
        bar_width = 0.24

        figure, axis = plt.subplots(figsize=(10.24, 7.68), dpi=100)
        for backend_index, backend in enumerate(backend_order):
            indexed = {
                (row["resolution"], row["operation"]): float(row["fps"])
                for row in results[backend]
                if int(row["p"]) == p
            }
            missing = [category for category in categories if category not in indexed]
            if missing:
                raise ValueError(f"{backend}.csv has incomplete p={p} results: {missing}")
            offset = (backend_index - 1) * bar_width
            axis.bar(
                x_positions + offset,
                [indexed[category] for category in categories],
                bar_width,
                label=f"{BACKEND_LABELS[backend]} FPS",
                color=BACKEND_COLORS[backend],
                edgecolor="none",
            )

        axis.set_xticks(x_positions, labels, rotation=48, ha="right", rotation_mode="anchor")
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f} fps"))
        axis.grid(axis="y", color="#c7d0d5", linewidth=0.8)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False, fontsize=13)
        axis.set_title(f"ME watermark throughput - window size p={p}", fontsize=14, pad=42)

        metadata = []
        for backend in backend_order:
            first = results[backend][0]
            metadata.append(f"{BACKEND_LABELS[backend]}: {first['device']} ({first['loops']} loops)")
        figure.text(0.99, 0.015, "\n".join(metadata), ha="right", va="bottom", fontsize=8, color="#33434d")
        figure.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
        output_path = BENCHMARK_DIR / f"{figure_number}.png"
        figure.savefig(output_path, facecolor="white")
        plt.close(figure)
        print(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true", help="run all three Release CLIs, then generate figures")
    action.add_argument("--figures", action="store_true", help="generate figures from existing CSV files only")
    parser.add_argument("--opencl-device-id", type=int, help="override settings.ini for the OpenCL benchmark")
    parser.add_argument("--loops", type=int, help="iterations per measurement for each backend (requires --run)")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        if arguments.loops is not None and (not arguments.run or arguments.loops <= 0):
            raise ValueError("--loops requires --run and a positive integer")
        if arguments.run:
            run_benchmarks(arguments.opencl_device_id, arguments.loops)
        generate_figures(read_results())
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
