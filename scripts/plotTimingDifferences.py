#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Plot every timing-difference array found in a barycentering result."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIELD_ORDER = (
    "total",
    "roemer",
    "roemer_geo",
    "roemer_obs",
    "shapiro",
    "einstein",
    "einstein_geo",
    "einstein_obs",
    "binary",
)


def resolve_input(path: Path) -> Path:
    if path.is_dir():
        path = path / "timing-components.npz"
    if not path.is_file():
        raise FileNotFoundError(f"result file not found: {path}")
    return path


def find_difference_fields(result) -> list[str]:
    prefix = "difference_"
    fields = [
        key.removeprefix(prefix) for key in result.files if key.startswith(prefix)
    ]
    order = {field: index for index, field in enumerate(FIELD_ORDER)}
    return sorted(fields, key=lambda field: (order.get(field, len(order)), field))


def display_scale(values: np.ndarray) -> tuple[float, str]:
    finite = np.abs(values[np.isfinite(values)])
    maximum = float(np.max(finite)) if finite.size else 0.0
    if maximum >= 1.0:
        return 1.0, "s"
    if maximum >= 1e-3:
        return 1e3, "ms"
    if maximum >= 1e-6:
        return 1e6, "µs"
    if maximum >= 1e-9:
        return 1e9, "ns"
    return 1e12, "ps"


def data_limits(
    values: np.ndarray, padding_fraction: float = 0.05
) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return -1.0, 1.0
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    span = upper - lower
    if span == 0.0:
        span = max(abs(lower), 1.0) * 0.1
    padding = padding_fraction * span
    return lower - padding, upper + padding


def plot_result(
    input_path: Path, output_path: Path, title: str | None, dpi: int
) -> None:
    with np.load(input_path) as result:
        if "t_gps" not in result:
            raise KeyError(f"{input_path} does not contain t_gps")
        fields = find_difference_fields(result)
        if not fields:
            raise KeyError(f"{input_path} contains no difference_* arrays")

        t_gps = np.asarray(result["t_gps"], dtype=float)
        days = (t_gps - t_gps[0]) / 86400.0
        columns = 2 if len(fields) > 1 else 1
        rows = math.ceil(len(fields) / columns)
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(6.4 * columns, 2.8 * rows),
            sharex=True,
            squeeze=False,
            constrained_layout=True,
        )

        for index, field in enumerate(fields):
            axis = axes.flat[index]
            values = np.atleast_2d(
                np.asarray(result[f"difference_{field}"], dtype=float)
            )
            if values.shape[1] != len(days):
                raise ValueError(
                    f"difference_{field} has {values.shape[1]} samples, "
                    f"but t_gps has {len(days)}"
                )
            scale, unit = display_scale(values)
            scaled_values = scale * values
            for point_index, series in enumerate(scaled_values):
                axis.plot(
                    days,
                    series,
                    linewidth=0.8,
                    label=f"point {point_index + 1}",
                )
            lower, upper = data_limits(scaled_values)
            axis.set_ylim(lower, upper)
            if lower <= 0.0 <= upper:
                axis.axhline(0.0, color="0.6", linewidth=0.5)
            axis.set_title(field.replace("_", " "))
            axis.set_ylabel(f"LAL - PINT [{unit}]")
            axis.grid(alpha=0.25)

        for axis in axes[-1, :]:
            axis.set_xlabel("days from GPS start")
        for index in range(len(fields), rows * columns):
            axes.flat[index].set_visible(False)

        first_values = np.atleast_2d(result[f"difference_{fields[0]}"])
        if first_values.shape[0] > 1:
            axes.flat[0].legend(loc="best", fontsize="small")
        figure.suptitle(title or input_path.parent.name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=dpi)
        plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="plot all difference_* fields in timing-components.npz"
    )
    parser.add_argument(
        "result",
        type=Path,
        help="result directory or path to timing-components.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output image path; defaults to RESULT/timing-differences.png",
    )
    parser.add_argument("--title", help="optional figure title")
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        input_path = resolve_input(args.result)
        output_path = args.output or input_path.with_name("timing-differences.png")
        plot_result(input_path, output_path, args.title, args.dpi)
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
