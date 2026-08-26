#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare LALPulsar barycentering delays with PINT.

This script uses only the public ``lal`` and ``lalpulsar`` SWIG modules.  It
requires the accompanying LALSuite patch, which adds detailed timing vectors
to ``lalpulsar.SSBtimes``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import lal
import lalpulsar as lp
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.time import Time
from pint.logging import setup as setup_pint_logging
from pint.models import get_model
from pint.toa import get_TOAs_array

setup_pint_logging(level="WARNING")

warnings.filterwarnings(
    "ignore",
    category=ResourceWarning,
    message=r"unclosed file <_io\.TextIOWrapper name='.*/\.astropy/cache/download/url/.*/contents'.*>",
)
warnings.filterwarnings(
    "ignore",
    category=ResourceWarning,
    message=r"unclosed <ssl\.SSLSocket .*>",
)

BASE_DELAY_FIELDS = (
    "total",
    "roemer",
    "roemer_geo",
    "roemer_obs",
    "shapiro",
    "einstein",
    "einstein_geo",
    "einstein_obs",
)


def timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def one_or_two(values: list[float], name: str) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2 and values[0] <= values[1]:
        return values[0], values[1]
    raise ValueError(f"{name} needs one value or an ordered MIN MAX pair")


def draw_uniform(rng: np.random.Generator, bounds: tuple[float, float], size: int):
    low, high = bounds
    if low == high:
        return np.full(size, low, dtype=float)
    return rng.uniform(low, high, size)


def make_timestamps(t_gps: np.ndarray):
    timestamps = lp.CreateTimestampVector(len(t_gps))
    for index, value in enumerate(t_gps):
        timestamps.data[index] = lal.LIGOTimeGPS(float(value))
    if len(t_gps) > 1:
        timestamps.deltaT = float(t_gps[1] - t_gps[0])
    return timestamps


def resolve_lalpulsar_data_file(name: str) -> str:
    path = lp.PulsarFileResolvePath(name)
    if not path:
        raise FileNotFoundError(
            f"LALPulsar could not resolve '{name}'; check LAL_DATA_PATH"
        )
    return path


def get_detector_states(
    t_gps: np.ndarray,
    detector: str,
    ephemeris: str,
    einstein: str,
):
    earth = resolve_lalpulsar_data_file(f"earth00-40-{ephemeris}.dat.gz")
    sun = resolve_lalpulsar_data_file(f"sun00-40-{ephemeris}.dat.gz")
    ephemeris_data = lp.InitBarycenter(earth, sun)

    timestamps = make_timestamps(t_gps)
    site = lp.GetSiteInfo(detector)
    states = lp.GetDetectorStates(timestamps, site, ephemeris_data, 0.0)

    # Public LALSuite's GetDetectorStates() uses its original analytic TT-TDB
    # correction.  Replace each EarthState through the public SWIG routine when
    # the numerical TEMPO2 correction is requested.
    if einstein == "tdb":
        time_file = resolve_lalpulsar_data_file("tdb_2000-2040.dat.gz")
        time_data = lp.InitTimeCorrections(time_file)
        for state in states.data:
            lp.BarycenterEarthNew(
                state.earthState,
                state.tGPS,
                ephemeris_data,
                time_data,
                lp.TIMECORRECTION_TDB,
            )
    elif einstein != "tdb_old":
        raise ValueError("--lal-einstein must be 'tdb_old' or 'tdb'")

    return states


def set_lal_tp(doppler, tp: float) -> None:
    # Public LALSuite uses LIGOTimeGPS here.  Newer development versions use a
    # REAL8, so accepting both keeps the driver useful with either build.
    if isinstance(doppler.tp, (float, np.floating)):
        doppler.tp = float(tp)
    else:
        doppler.tp = lal.LIGOTimeGPS(float(tp))


def make_lal_doppler(params: dict[str, float], reference_gps: float):
    doppler = lp.PulsarDopplerParams()
    doppler.Alpha = params["alpha"]
    doppler.Delta = params["delta"]
    doppler.refTime = lal.LIGOTimeGPS(reference_gps)
    doppler.fkdot[0] = params["frequency"]
    doppler.asini = params["asini"]
    doppler.period = params["period"]
    set_lal_tp(doppler, params["tp"])
    doppler.ecc = params["ecc"]
    doppler.argp = params["argp"]
    if hasattr(doppler, "orbRefPoint"):
        doppler.orbRefPoint = int(getattr(lp, "ORBITAL_REFERENCE_POINT_PERI", 0))
    return doppler


def make_pint_model(is_binary: bool, binary_model: str):
    model_text = """
        RAJ             0
        DECJ            0
        PEPOCH          55000
        F0              100
        UNITS           TDB
        DM              0
    """
    if is_binary:
        model_text += f"""
        BINARY          {binary_model}
        A1              1.0
        PB              1.0
        T0              55000
        ECC             0.0
        OM              0.0
        """
    return get_model(io.StringIO(model_text))


def set_pint_model(model, params: dict[str, float], is_binary: bool) -> None:
    model["RAJ"].quantity = Angle(params["alpha"], unit="radian")
    model["DECJ"].quantity = Angle(params["delta"], unit="radian")
    model["F0"].quantity = params["frequency"] * u.Hz
    if not is_binary:
        return

    model["A1"].quantity = params["asini"] * u.lightsecond
    model["PB"].quantity = params["period"] * u.s
    tp_mjd_tai = Time(
        Time(params["tp"], format="gps").mjd,
        format="mjd",
        scale="tai",
    )
    # LAL interprets tp as GPS(TDB).  Only shift the numeric origin to MJD;
    # PINT must not apply another conversion to the supplied epoch.
    model["T0"].value = tp_mjd_tai.tt.mjd
    model["ECC"].quantity = params["ecc"] * u.dimensionless_unscaled
    model["OM"].quantity = Angle(params["argp"], unit="radian")


def sample_parameters(args) -> list[dict[str, float]]:
    rng = np.random.default_rng(args.seed)

    if args.sky_position:
        if args.alpha is not None or args.delta is not None:
            raise ValueError(
                "--sky-position cannot be combined with --alpha or --delta"
            )
        count = len(args.sky_position)
        alpha = np.array([position[0] for position in args.sky_position])
        delta = np.array([position[1] for position in args.sky_position])
    else:
        count = args.num_doppler
        if (args.alpha is None) != (args.delta is None):
            raise ValueError("--alpha and --delta must be supplied together")
        if args.alpha is None:
            alpha = rng.uniform(0.0, 2.0 * np.pi, count)
            delta = np.arcsin(rng.uniform(-1.0, 1.0, count))
        else:
            alpha = np.full(count, args.alpha)
            delta = np.full(count, args.delta)

    asini = draw_uniform(rng, one_or_two(args.asini, "--asini"), count)
    period = draw_uniform(rng, one_or_two(args.period, "--period"), count)
    ecc = draw_uniform(rng, one_or_two(args.ecc, "--ecc"), count)

    if args.argp is None:
        argp = (
            rng.uniform(0.0, 2.0 * np.pi, count) if np.any(ecc > 0) else np.zeros(count)
        )
    else:
        argp = draw_uniform(rng, one_or_two(args.argp, "--argp"), count)

    if args.tp is None:
        tp = np.array(
            [rng.uniform(0.0, value) if value > 0 else 0.0 for value in period]
        )
    else:
        tp = draw_uniform(rng, one_or_two(args.tp, "--tp"), count)

    if np.any(asini < 0):
        raise ValueError("--asini must be non-negative")
    if np.any((ecc < 0) | (ecc >= 1)):
        raise ValueError("--ecc must satisfy 0 <= ecc < 1")
    if np.any((asini > 0) & (period <= 0)):
        raise ValueError("binary points need a positive --period")

    parameters = []
    for index in range(count):
        if asini[index] > 0:
            vp_over_c = (
                2.0
                * np.pi
                * asini[index]
                / period[index]
                * math.sqrt((1.0 + ecc[index]) / (1.0 - ecc[index]))
            )
            if vp_over_c >= 1.0:
                raise ValueError(f"sample {index} has projected periapsis speed >= c")
        parameters.append(
            {
                "alpha": float(alpha[index]),
                "delta": float(delta[index]),
                "frequency": float(args.frequency),
                "asini": float(asini[index]),
                "period": float(period[index]),
                "tp": float(tp[index]),
                "ecc": float(ecc[index]),
                "argp": float(argp[index]),
            }
        )
    return parameters


def vector_data(ssb, field: str) -> np.ndarray:
    vector = getattr(ssb, field, None)
    if vector is None:
        raise RuntimeError(
            f"patched LALSuite did not return SSBtimes.{field}; apply the supplied patch and rebuild SWIG"
        )
    return np.array(vector.data, dtype=float, copy=True)


def compute(args):
    if not hasattr(lp, "SSBPREC_RELATIVISTICOPT_DETAILED"):
        raise RuntimeError(
            "lalpulsar lacks SSBPREC_RELATIVISTICOPT_DETAILED; apply the supplied LALSuite patch"
        )

    t_gps = np.arange(args.gps_start, args.gps_start + args.duration, args.cadence)
    if len(t_gps) < 3:
        raise ValueError(
            "the requested interval must contain at least three timestamps"
        )
    reference_gps = (
        args.reference_gps
        if args.reference_gps is not None
        else args.gps_start + 0.5 * args.duration
    )

    parameters = sample_parameters(args)
    is_binary = any(point["asini"] > 0 for point in parameters)
    fields = BASE_DELAY_FIELDS + (("binary",) if is_binary else ())

    t_mjd = Time(Time(t_gps, format="gps").mjd, scale="tai", format="mjd")
    toas, runtime_toas = timed_call(
        get_TOAs_array,
        t_mjd,
        obs=args.detector,
        ephem=args.pint_ephemeris,
        include_bipm=args.pint_use_bipm,
    )
    toas_geo = get_TOAs_array(
        t_mjd,
        obs="geocenter",
        ephem=args.pint_ephemeris,
        include_bipm=args.pint_use_bipm,
    )

    states, runtime_states = timed_call(
        get_detector_states,
        t_gps,
        args.detector,
        args.lal_ephemeris,
        args.lal_einstein,
    )

    model = make_pint_model(is_binary, args.pint_binary_model)
    arrays: dict[str, list[np.ndarray] | np.ndarray] = {
        "t_gps": t_gps,
        "alpha": np.array([point["alpha"] for point in parameters]),
        "delta": np.array([point["delta"] for point in parameters]),
        "frequency": np.array([point["frequency"] for point in parameters]),
        "asini": np.array([point["asini"] for point in parameters]),
        "period": np.array([point["period"] for point in parameters]),
        "tp": np.array([point["tp"] for point in parameters]),
        "ecc": np.array([point["ecc"] for point in parameters]),
        "argp": np.array([point["argp"] for point in parameters]),
        "lp_tdot": [],
        "rel_impact_sun": [],
    }
    for prefix in ("lp", "pint"):
        for field in fields:
            arrays[f"{prefix}_{field}"] = []

    t_tt = t_mjd.tt
    t_tdb = toas.table["tdb"]
    t_tdb_geo = toas_geo.table["tdb"]
    pint_einstein = 86400.0 * np.array(
        [(tt.jd1 - tdb.jd1) + (tt.jd2 - tdb.jd2) for tt, tdb in zip(t_tt, t_tdb)]
    )
    pint_einstein_geo = 86400.0 * np.array(
        [(tt.jd1 - tdb.jd1) + (tt.jd2 - tdb.jd2) for tt, tdb in zip(t_tt, t_tdb_geo)]
    )
    pint_einstein_obs = pint_einstein - pint_einstein_geo

    runtime_lal_ssb = []
    runtime_pint_delay = []
    for point in parameters:
        sky = lal.SkyPosition()
        sky.longitude = point["alpha"]
        sky.latitude = point["delta"]
        sky.system = lal.COORDINATESYSTEM_EQUATORIAL

        ssb, elapsed = timed_call(
            lp.GetSSBtimes,
            states,
            sky,
            lal.LIGOTimeGPS(reference_gps),
            lp.SSBPREC_RELATIVISTICOPT_DETAILED,
        )
        runtime_lal_ssb.append(elapsed)
        if is_binary:
            doppler = make_lal_doppler(point, reference_gps)
            lp.AddBinaryTimes(ssb, ssb, doppler)

        arrays["lp_total"].append(vector_data(ssb, "total"))
        arrays["lp_tdot"].append(vector_data(ssb, "Tdot"))
        arrays["lp_roemer_geo"].append(vector_data(ssb, "roemer_geo"))
        arrays["lp_roemer_obs"].append(vector_data(ssb, "roemer_obs"))
        arrays["lp_roemer"].append(
            arrays["lp_roemer_geo"][-1] + arrays["lp_roemer_obs"][-1]
        )
        arrays["lp_einstein_geo"].append(vector_data(ssb, "einstein_geo"))
        arrays["lp_einstein_obs"].append(vector_data(ssb, "einstein_obs"))
        arrays["lp_einstein"].append(
            arrays["lp_einstein_geo"][-1] + arrays["lp_einstein_obs"][-1]
        )
        arrays["lp_shapiro"].append(vector_data(ssb, "shapiro"))
        arrays["rel_impact_sun"].append(vector_data(ssb, "rel_impact_sun"))
        if is_binary:
            arrays["lp_binary"].append(vector_data(ssb, "binary"))

        set_pint_model(model, point, is_binary)
        pint_delay, elapsed = timed_call(model.delay, toas=toas)
        runtime_pint_delay.append(elapsed)

        pint_roemer = np.asarray(model.solar_system_geometric_delay(toas).value)
        pint_roemer_geo = np.asarray(model.solar_system_geometric_delay(toas_geo).value)
        arrays["pint_roemer"].append(pint_roemer)
        arrays["pint_roemer_geo"].append(pint_roemer_geo)
        arrays["pint_roemer_obs"].append(pint_roemer - pint_roemer_geo)
        arrays["pint_einstein"].append(pint_einstein.copy())
        arrays["pint_einstein_geo"].append(pint_einstein_geo.copy())
        arrays["pint_einstein_obs"].append(pint_einstein_obs.copy())
        arrays["pint_shapiro"].append(
            np.asarray(model.solar_system_shapiro_delay(toas).value)
        )
        if is_binary:
            arrays["pint_binary"].append(
                np.asarray(model.binarymodel_delay(toas).value)
            )
        # PINT's model delay is relative to TDB, so add TT-TDB explicitly.
        arrays["pint_total"].append(np.asarray(pint_delay.value) + pint_einstein)

    for key, value in tuple(arrays.items()):
        if isinstance(value, list):
            arrays[key] = np.atleast_2d(value).astype(float)

    if args.avoid_sun:
        mask = (arrays["rel_impact_sun"] >= 0) & (arrays["rel_impact_sun"] <= 1)
        for field in ("total", "shapiro"):
            arrays[f"lp_{field}"][mask] = np.nan
            arrays[f"pint_{field}"][mask] = np.nan

    summary = {}
    for field in fields:
        difference = arrays[f"lp_{field}"] - arrays[f"pint_{field}"]
        arrays[f"difference_{field}"] = difference
        std = np.nanstd(difference, axis=1)
        arrays[f"std_{field}"] = std
        if np.isnan(std).any():
            raise RuntimeError(f"std_{field} contains NaN values")
        summary[f"std_{field}"] = {
            "min": float(np.min(std)),
            "mean": float(np.mean(std)),
            "max": float(np.max(std)),
        }

    numerical_tdot = 1.0 - np.gradient(arrays["lp_total"], t_gps, axis=1, edge_order=2)
    arrays["numerical_tdot"] = numerical_tdot
    relative_tdot_error = np.abs(numerical_tdot - arrays["lp_tdot"]) / np.abs(
        arrays["lp_tdot"]
    )
    arrays["relative_tdot_error"] = relative_tdot_error
    interior_error = relative_tdot_error[:, 1:-1]
    summary["tdot_relative_error"] = {
        "min": float(np.nanmin(interior_error)),
        "mean": float(np.nanmean(interior_error)),
        "max": float(np.nanmax(interior_error)),
    }

    timings = {
        "lal_GetDetectorStates": runtime_states,
        "pint_get_TOAs_array": runtime_toas,
        "lal_GetSSBtimes_mean": float(np.mean(runtime_lal_ssb)),
        "pint_delay_mean": float(np.mean(runtime_pint_delay)),
    }
    metadata = {
        "command": sys.argv,
        "detector": args.detector,
        "gps_start": args.gps_start,
        "duration": args.duration,
        "cadence": args.cadence,
        "reference_gps": reference_gps,
        "num_timestamps": len(t_gps),
        "num_doppler": len(parameters),
        "lal_ephemeris": args.lal_ephemeris,
        "lal_einstein": args.lal_einstein,
        "pint_ephemeris": args.pint_ephemeris,
        "pint_use_bipm": args.pint_use_bipm,
        "pint_binary_model": args.pint_binary_model if is_binary else None,
        "avoid_sun": args.avoid_sun,
        "is_binary": is_binary,
        "versions": {
            "lal": getattr(lal, "__version__", "unknown"),
            "lalpulsar": getattr(lp, "__version__", "unknown"),
            "pint-pulsar": package_version("pint-pulsar"),
            "astropy": package_version("astropy"),
            "numpy": np.__version__,
        },
    }
    return arrays, summary, timings, metadata


def parse_thresholds(items: list[str]) -> dict[str, float]:
    thresholds = {}
    for item in items:
        try:
            field, value = item.split("=", 1)
            thresholds[field] = float(value)
        except ValueError as exc:
            raise ValueError(
                f"invalid --std-max '{item}', expected FIELD=SECONDS"
            ) from exc
    return thresholds


def check_thresholds(summary, std_thresholds, tdot_threshold):
    for field, limit in std_thresholds.items():
        key = f"std_{field}"
        if key not in summary:
            raise ValueError(f"unknown --std-max field '{field}'")
        observed = summary[key]["max"]
        if observed > limit:
            raise RuntimeError(f"maximum {key} {observed:.3g} s exceeds {limit:.3g} s")
    observed = summary["tdot_relative_error"]["max"]
    if tdot_threshold is not None and observed > tdot_threshold:
        raise RuntimeError(
            f"maximum Tdot relative error {observed:.3g} exceeds {tdot_threshold:.3g}"
        )


def report(summary):
    width = max(len(key) for key in summary)
    for key, stats in summary.items():
        unit = "" if key == "tdot_relative_error" else " s"
        print(
            f"{key:<{width}} : min={stats['min']:#.3g}{unit}  "
            f"mean={stats['mean']:#.3g}{unit}  max={stats['max']:#.3g}{unit}"
        )


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="compare patched public LALSuite barycentering with PINT"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("barycentering-results")
    )
    parser.add_argument("--gps-start", type=float, default=1368921618.0)
    parser.add_argument("--duration", type=float, default=86400.0)
    parser.add_argument("--cadence", type=float, default=3600.0)
    parser.add_argument("--reference-gps", type=float)
    parser.add_argument("--detector", default="L1")
    parser.add_argument("--lal-ephemeris", default="DE430")
    parser.add_argument("--pint-ephemeris", default="DE430")
    parser.add_argument("--lal-einstein", choices=("tdb_old", "tdb"), default="tdb")
    parser.add_argument("--pint-use-bipm", action="store_true")
    parser.add_argument("--pint-binary-model", choices=("BT", "DD"), default="DD")
    parser.add_argument("--num-doppler", type=int, default=1)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--alpha", type=float, help="fixed right ascension in radians")
    parser.add_argument("--delta", type=float, help="fixed declination in radians")
    parser.add_argument(
        "--sky-position",
        action="append",
        nargs=2,
        type=float,
        metavar=("ALPHA", "DELTA"),
        help="explicit sky position in radians; repeat for multiple positions",
    )
    parser.add_argument("--frequency", type=float, default=100.0)
    parser.add_argument(
        "--asini", type=float, nargs="+", default=[0.0], metavar="VALUE"
    )
    parser.add_argument(
        "--period", type=float, nargs="+", default=[0.0], metavar="VALUE"
    )
    parser.add_argument("--tp", type=float, nargs="+", metavar="VALUE")
    parser.add_argument("--ecc", type=float, nargs="+", default=[0.0], metavar="VALUE")
    parser.add_argument("--argp", type=float, nargs="+", metavar="VALUE")
    parser.add_argument(
        "--avoid-sun", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--std-max",
        action="append",
        default=[],
        metavar="FIELD=SECONDS",
        help="fail if the maximum per-sky standard deviation exceeds this value",
    )
    parser.add_argument("--tdot-relerr-max", type=float)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.duration <= 0 or args.cadence <= 0:
        parser.error("--duration and --cadence must be positive")
    if args.num_doppler < 1:
        parser.error("--num-doppler must be positive")

    try:
        arrays, summary, timings, metadata = compute(args)
        check_thresholds(
            summary,
            parse_thresholds(args.std_max),
            args.tdot_relerr_max,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "timing-components.npz", **arrays)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "timings.json", timings)
    write_json(args.output_dir / "metadata.json", metadata)
    report(summary)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
