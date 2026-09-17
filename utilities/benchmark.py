#!/usr/bin/env python
"""
Benchmark
=========

Cross-backend benchmarks for *Colour*, covering correctness and performance
across *Array API* backends (*NumPy*, *JAX*, *PyTorch CPU*, *PyTorch MPS*).

Each benchmark suite is a concrete subclass of :class:`BenchmarkSuite` and
emits :class:`BenchmarkResult` records accumulated in a
:class:`BenchmarkReport`.
A :class:`BenchmarkRunner` orchestrates shared execution concerns (backend
dispatch, warmup / timed runs, memory release).

Currently implemented suites:

-   :class:`BenchmarkSuite_ConversionGraph`, walks the array-path of the
    *Colour* automatic conversion graph (iterative methods filtered out
    via :attr:`BenchmarkConfiguration.iterative_graph_cases`).
-   :class:`BenchmarkSuite_ConversionGraphIterative`, walks the same
    graph but yields only methods backed by ``solve_CCT_Newton`` or
    ``scipy.spatial.distance.cdist``; measures Python-loop overhead.
-   :class:`BenchmarkSuite_Difference`, colour difference functions
    (Delta E family).
-   :class:`BenchmarkSuite_IntegrationArray`, spectral integration
    array-path (``sd_to_XYZ`` / ``msds_to_XYZ`` with raw ``ArrayLike``
    inputs and ``method="Integration"``); dispatches through the *Array
    API*.
-   :class:`BenchmarkSuite_IntegrationObject`, spectral integration
    object-path (``SpectralDistribution`` / ``MultiSpectralDistributions``
    inputs across all methods); numpy-bound by the SD object machinery.
-   :class:`BenchmarkSuite_TransferFunction`, CCTF encoding / decoding.
-   :class:`BenchmarkSuite_Adaptation`, direct chromatic adaptation
    transforms.
-   :class:`BenchmarkSuite_Characterisation`, colour correction matrix
    methods.
-   :class:`BenchmarkSuite_RecoveryArray`, vectorisable spectral
    reflectance recovery (*Smits 1999*, *Gaussian*, *Mallett 2019*,
    *Jakob 2019* LUT runtime).
-   :class:`BenchmarkSuite_RecoveryObject`, per-pixel spectral
    reflectance recovery solvers (*Jakob 2019*, *Meng 2015*, *Otsu 2018*).
-   :class:`BenchmarkSuite_QualityArray`, batched
    :class:`MultiSpectralDistributions` paths for *CRI* / *CFI* / *CQS*;
    captures the win from computing the *Planckian* / *CIE D Series*
    references via the array kernels (``planck_law``,
    ``CIE_illuminant_D_series``) rather than per-entry
    :class:`MultiSpectralDistributions` construction.
-   :class:`BenchmarkSuite_QualityObject`, light-source quality indices
    (CRI / CFI / CQS / SSI) on single :class:`SpectralDistribution`
    inputs; numpy-bound by the SD object machinery.
-   :class:`BenchmarkSuite_Volume`, gamut volume on ``RGB_Colourspace``
    instances (``Monte Carlo``, ``RGB_colourspace_limits``).
-   :class:`BenchmarkSuite_VolumeIterative`, ``scipy.spatial.Delaunay``-
    backed gamut containment checks (``is_within_visible_spectrum``,
    ``is_within_macadam_limits``).
-   :class:`BenchmarkSuite_Phenomena`, sky models and Rayleigh scattering.
-   :class:`BenchmarkSuite_TemperatureArray`, closed-form *CIE xy*
    <-> ``CCT`` methods that dispatch as one-shot *Array API* operations.
-   :class:`BenchmarkSuite_TemperatureIterative`, *CIE xy* <-> ``CCT``
    methods backed by :func:`solve_CCT_Newton` /
    :func:`solve_xy_Newton`; iteration cost scales with batch size.
-   :class:`BenchmarkSuite_Blindness`, colour-vision deficiency models.
-   :class:`BenchmarkSuite_Contrast`, contrast sensitivity functions.
-   :class:`BenchmarkSuite_GeneratorsArray`, vectorised spectral
    generators (``planck_law``, ``rayleigh_jeans_law``); array-path,
    dispatches through the *Array API*.
-   :class:`BenchmarkSuite_GeneratorsObject`, per-:class:`SpectralDistribution`
    generators (``sd_blackbody``, ``sd_rayleigh_jeans``,
    ``sd_CIE_illuminant_D_series``, ``msds_*`` variants); numpy-bound
    by the SD object machinery.
-   :class:`BenchmarkSuite_Photometry`, photometric measures over a
    :class:`SpectralDistribution` (``luminous_flux``,
    ``luminous_efficacy``, ``luminous_efficiency``); numpy-bound.

Usage::

    uv run python utilities/benchmark.py --mode quick
    uv run python utilities/benchmark.py --mode full
    uv run python utilities/benchmark.py --mode full \\
        --suites conversion_graph,difference
    uv run python utilities/benchmark.py --mode full \\
        --json results.json --csv results.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import logging
import statistics
import sys
import time
import traceback
import typing
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field, fields, is_dataclass, replace
from functools import cached_property, partial

import numpy as np

from colour.adaptation import CHROMATIC_ADAPTATION_METHODS, chromatic_adaptation
from colour.algebra import vecmul
from colour.appearance import (
    CAM_Specification_CAM16,
    CAM_Specification_CIECAM02,
    CAM_Specification_CIECAM16,
    CAM_Specification_Hellwig2022,
    CAM_Specification_Kim2009,
    CAM_Specification_sCAM,
    CAM_Specification_ZCAM,
)
from colour.blindness import matrix_cvd_Machado2009
from colour.characterisation import (
    MATRIX_COLOUR_CORRECTION_METHODS,
    apply_matrix_colour_correction,
    matrix_colour_correction,
)
from colour.colorimetry import (
    LIGHTNESS_METHODS,
    LUMINANCE_METHODS,
    MSDS_CMFS,
    MSDS_TO_XYZ_METHODS,
    SD_TO_XYZ_METHODS,
    SDS_ILLUMINANTS,
    SDS_LIGHT_SOURCES,
    WHITENESS_METHODS,
    YELLOWNESS_METHODS,
    CIE_illuminant_D_series,
    MultiSpectralDistributions,
    SpectralDistribution,
    SpectralShape,
    luminous_efficacy,
    luminous_efficiency,
    luminous_flux,
    msds_blackbody,
    msds_CIE_illuminant_D_series,
    msds_rayleigh_jeans,
    msds_to_XYZ,
    planck_law,
    rayleigh_jeans_law,
    sd_blackbody,
    sd_CIE_illuminant_D_series,
    sd_rayleigh_jeans,
    sd_to_XYZ,
)
from colour.contrast import contrast_sensitivity_function
from colour.difference import DELTA_E_METHODS
from colour.graph.conversion import CONVERSION_SPECIFICATIONS_DATA, convert
from colour.models import (
    CCTF_DECODINGS,
    CCTF_ENCODINGS,
    RGB_COLOURSPACES,
    RGB_COLOURSPACE_sRGB,
)
from colour.notation import MUNSELL_VALUE_METHODS
from colour.phenomena import (
    rayleigh_optical_depth,
    sd_rayleigh_scattering,
    sky_luminance_distribution_CIE2003,
    sky_luminance_distribution_overcast_CIE2003,
    sky_scattering_indicatrix_CIE2003,
)
from colour.quality import (
    colour_fidelity_index,
    colour_quality_scale,
    colour_rendering_index,
    spectral_similarity_index,
)
from colour.recovery import (
    LUT3D_Jakob2019,
    RGB_to_msds_Smits1999,
    XYZ_to_sd,
    generate_gaussian_basis,
)
from colour.recovery.jakob2019 import SPECTRAL_SHAPE_JAKOB2019
from colour.recovery.mallett2019 import MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019
from colour.temperature import (
    CCT_TO_UV_METHODS,
    UV_TO_CCT_METHODS,
    CCT_to_xy,
    xy_to_CCT,
)
from colour.utilities import array_api_enable, set_default_float_dtype
from colour.volume import (
    RGB_colourspace_limits,
    RGB_colourspace_volume_MonteCarlo,
    is_within_macadam_limits,
    is_within_visible_spectrum,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

jax: typing.Any = None
jnp: typing.Any = None
with contextlib.suppress(ImportError):
    import jax
    import jax.numpy as jnp

torch: typing.Any = None
with contextlib.suppress(ImportError):
    import torch

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "BenchmarkConfiguration",
    "DEFAULT_BENCHMARK_CONFIGURATION",
    "BenchmarkResult",
    "BackendStatistics",
    "BenchmarkReport",
    "write_report_json",
    "write_report_csv",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "BenchmarkSuite_ConversionGraph",
    "BenchmarkSuite_ConversionGraphIterative",
    "BenchmarkSuite_Difference",
    "BenchmarkSuite_IntegrationArray",
    "BenchmarkSuite_IntegrationObject",
    "BenchmarkSuite_TransferFunction",
    "BenchmarkSuite_Adaptation",
    "BenchmarkSuite_Characterisation",
    "BenchmarkSuite_RecoveryArray",
    "BenchmarkSuite_RecoveryObject",
    "BenchmarkSuite_QualityArray",
    "BenchmarkSuite_QualityObject",
    "BenchmarkSuite_Volume",
    "BenchmarkSuite_VolumeIterative",
    "BenchmarkSuite_Phenomena",
    "BenchmarkSuite_TemperatureArray",
    "BenchmarkSuite_TemperatureIterative",
    "BenchmarkSuite_Blindness",
    "BenchmarkSuite_Contrast",
    "BenchmarkSuite_GeneratorsArray",
    "BenchmarkSuite_GeneratorsObject",
    "BenchmarkSuite_Photometry",
    "BENCHMARK_SUITES",
    "build_argument_parser",
    "main",
]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkConfiguration:
    """
    Define static configuration for the benchmark suites.

    Parameters
    ----------
    backends
        Canonical backend ordering for reporting.
    input_resolutions
        Pixel counts keyed by input size tag.
    skip_sources
        Conversion graph source nodes that cannot be exercised with plain
        array inputs.
    skip_targets
        Conversion graph target nodes that produce non-array outputs.
    cam_sources
        Colour appearance model sources requiring dataclass inputs (i.e. an
        instance of :class:`colour.appearance.CAM_Specification_*`).
    reduced_size_targets
        Conversion graph targets allocating ``O(N * M)`` intermediates;
        reduced pixel count is used to avoid OOM errors.
    reduced_size_edges
        Conversion graph edges backed by iterative solvers that do not
        scale to HD inputs.
    skip_mps_edges
        Conversion graph edges that crash the *PyTorch MPS* runtime.
    exception_categories
        Ordered ``(code, substrings)`` pairs used by
        :meth:`BenchmarkResult.categorise_exception`.
    cam_specifications
        Ordered ``(source, specification)`` pairs mapping CAM source names to
        their specification dataclasses.
    edge_method_registries
        Conversion graph edges whose ``convert(..., method=...)`` kwarg fans
        out across a method registry, yielding one benchmark case per method.
    rng_seed
        Seed used to build deterministic :class:`numpy.random.Generator`
        instances for reproducible benchmark inputs.

    Methods
    -------
    -   :meth:`~BenchmarkConfiguration.resolution`
    -   :meth:`~BenchmarkConfiguration.rng`
    """

    backends: tuple[str, ...] = ("numpy", "jax", "torch-cpu", "torch-mps")

    input_resolutions: tuple[tuple[str, int], ...] = (
        ("small", 2),
        ("reduced", 10000),
        ("hd", 1920 * 1080),
    )

    skip_sources: frozenset[str] = frozenset(
        {
            "Spectral Distribution",
            "Hexadecimal",
            "Munsell Colour",
            "CSS Color 3",
            "Wavelength",
        }
    )

    skip_targets: frozenset[str] = frozenset(
        {
            "Hexadecimal",
            "Munsell Colour",
            "Spectral Distribution",
            "Complementary Wavelength",
            "Dominant Wavelength",
            "Luminous Flux",
            "Luminous Efficiency",
            "Luminous Efficacy",
        }
    )

    cam_sources: frozenset[str] = frozenset(
        {
            "CIECAM02",
            "CAM16",
            "CIECAM16",
            "Hellwig 2022",
            "Kim 2009",
            "sCAM",
            "ZCAM",
        }
    )

    reduced_size_targets: frozenset[str] = frozenset(
        {
            "Colorimetric Purity",
            "Excitation Purity",
        }
    )

    reduced_size_edges: frozenset[tuple[str, str]] = frozenset(
        {
            ("OSA UCS", "CIE XYZ"),
            ("CCT", "CIE UCS uv"),
            ("CIE UCS uv", "CCT"),
        }
    )

    skip_mps_edges: frozenset[tuple[str, str]] = frozenset(
        {
            ("OSA UCS", "CIE XYZ"),
        }
    )

    exception_categories: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("DEV", ("device",)),
        ("TYP", ("must be tensor", "must be torch")),
        ("NIM", ("not implemented",)),
        ("OVF", ("overflow",)),
    )

    cam_specifications: tuple[tuple[str, type], ...] = (
        ("CAM16", CAM_Specification_CAM16),
        ("CIECAM02", CAM_Specification_CIECAM02),
        ("CIECAM16", CAM_Specification_CIECAM16),
        ("Hellwig 2022", CAM_Specification_Hellwig2022),
        ("Kim 2009", CAM_Specification_Kim2009),
        ("sCAM", CAM_Specification_sCAM),
        ("ZCAM", CAM_Specification_ZCAM),
    )

    edge_method_registries: tuple[tuple[tuple[str, str], typing.Any], ...] = (
        (("CIE XYZ", "Whiteness"), WHITENESS_METHODS),
        (("CIE XYZ", "Yellowness"), YELLOWNESS_METHODS),
        (("Luminance", "Lightness"), LIGHTNESS_METHODS),
        (("Lightness", "Luminance"), LUMINANCE_METHODS),
        (("Luminance", "Munsell Value"), MUNSELL_VALUE_METHODS),
        (("CIE UCS uv", "CCT"), UV_TO_CCT_METHODS),
        (("CCT", "CIE UCS uv"), CCT_TO_UV_METHODS),
    )

    iterative_graph_cases: frozenset[tuple[str, str, str | None]] = frozenset(
        {
            # ``CIE UCS uv -> CCT`` methods backed by
            # :func:`colour.temperature.solve_CCT_Newton`. The Newton
            # iteration itself is vectorised across samples but runs
            # ``newton_iterations * backtrack_iterations`` forward
            # evaluations per call (each of which is a full *Planckian*
            # SD generation at HD), so the cost is unrelated to the
            # closed-form Array-API path. The other ``uv -> CCT``
            # methods (``Robertson 1968`` isotherm-line broadcast,
            # ``Ohno 2013`` batched LUT scan) are fully vectorised and
            # stay in the array path.
            ("CIE UCS uv", "CCT", "Planck 1900"),
            ("CIE UCS uv", "CCT", "Krystek 1985"),
            # Per-sample spectral-locus search via
            # ``scipy.spatial.distance.cdist`` (forced host round-trip,
            # no Array-API dispatch possible).
            ("CIE xy", "Colorimetric Purity", None),
            ("CIE xy", "Excitation Purity", None),
        }
    )
    """
    ``(source, target, method_or_None)`` triples whose underlying
    implementation is iterative (*solve_CCT_Newton*) or *scipy*-bound
    (``scipy.spatial.distance.cdist``). The other slow rows in
    ``conversion_graph`` (``Ohno 2013``, all ``Munsell Value`` methods
    including ``ASTM D1535`` / ``McCamy 1987``) are fully vectorised
    closed-form / batched-LUT and stay in the array path.
    """

    rng_seed: int = 16

    def resolution(self, size: str) -> int:
        """Return the pixel count for a size tag (``small``/``reduced``/``hd``)."""

        return dict(self.input_resolutions)[size]

    def rng(self) -> np.random.Generator:
        """Return a fresh deterministic random generator seeded with ``rng_seed``."""

        return np.random.default_rng(self.rng_seed)


DEFAULT_BENCHMARK_CONFIGURATION = BenchmarkConfiguration()
"""Default configuration shared across the benchmark suites."""


@dataclass
class BenchmarkResult:
    """
    Define the outcome of benchmarking a single case on a single backend.

    Parameters
    ----------
    suite
        Name of the suite that produced the result.
    backend
        Backend identifier (``"numpy"``, ``"jax"``, ``"torch-cpu"``,
        ``"torch-mps"``).
    status
        One of ``"SUCCEEDED"``, ``"SKIPPED"``, or a short error code from
        :meth:`BenchmarkResult.categorise_exception`.
    label
        Human-readable identifier for the case, supplied by the suite.
    duration
        Best (minimum) of timed runs, in seconds.
    error
        Error message when :attr:`status` is not ``"SUCCEEDED"``. Stored in
        full: backend errors carry their diagnosis at the end of the message,
        e.g. which two devices mismatched, so truncating at capture time
        discards exactly what the failure is triaged with.
    traceback
        Formatted traceback when :attr:`status` is not ``"SUCCEEDED"``.
    metadata
        BenchmarkSuite-specific fields (e.g., ``source`` / ``target``, ``method``).

    Methods
    -------
    -   :meth:`~BenchmarkResult.format_duration`
    -   :meth:`~BenchmarkResult.categorise_exception`
    -   :meth:`~BenchmarkResult.failed`
    """

    suite: str
    backend: str
    status: str
    label: str = ""
    duration: float = 0.0
    error: str = ""
    traceback: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def format_duration(duration: float) -> str:
        """
        Format a duration given in seconds with an appropriate unit.

        Picks between ``s``, ``ms``, and ``us`` based on magnitude.
        """

        if duration >= 1.0:
            return f"{duration:.2f} s"

        milliseconds = duration * 1e3
        if milliseconds >= 10:
            return f"{round(milliseconds)} ms"
        if milliseconds >= 1:
            return f"{milliseconds:.1f} ms"
        if milliseconds >= 0.001:
            return f"{milliseconds:.2f} ms"

        return f"{milliseconds * 1e3:.2f} us"

    @staticmethod
    def categorise_exception(exception: Exception) -> str:
        """
        Return a short status code characterising ``exception``.

        Codes: ``DEV`` (device mismatch), ``TYP`` (type mismatch), ``NIM``
        (not implemented), ``OVF`` (overflow), ``OTH`` (other).
        """

        message = str(exception).lower()
        for code, substrings in DEFAULT_BENCHMARK_CONFIGURATION.exception_categories:
            if any(s in message for s in substrings):
                return code
        return "OTH"

    @classmethod
    def failed(
        cls,
        suite: str,
        backend: str,
        label: str,
        exception: Exception,
        metadata: dict[str, str],
    ) -> BenchmarkResult:
        """Build a failed-case :class:`BenchmarkResult` and collect memory."""

        gc.collect()
        return cls(
            suite=suite,
            backend=backend,
            status=cls.categorise_exception(exception),
            label=label,
            error=str(exception),
            traceback="".join(traceback.format_exception(exception)),
            metadata=metadata,
        )


@dataclass
class BackendStatistics:
    """
    Define aggregated statistics for one backend within one suite.

    Parameters
    ----------
    backend
        Backend identifier.
    succeeded
        Number of cases that completed successfully.
    skipped
        Number of cases skipped (e.g., known MPS segfaults).
    failed
        Number of cases that raised an exception.
    duration
        Median duration of successful cases, in seconds.
    speedup
        Geometric mean of per-case speedups vs the *NumPy* baseline.

    Methods
    -------
    -   :meth:`~BackendStatistics.median`
    -   :meth:`~BackendStatistics.geometric_mean`
    """

    backend: str
    succeeded: int
    skipped: int
    failed: int
    duration: float
    speedup: float

    @staticmethod
    def median(values: list[float]) -> float:
        """Return the median of ``values`` (``0`` for the empty list)."""

        return statistics.median(values) if values else 0.0

    @staticmethod
    def best(values: list[float]) -> float:
        """
        Return the minimum of ``values`` (``0`` for the empty list).

        Best-of-N is the conventional aggregator for micro-benchmarks: noise
        adds to the timing (cache misses, GC, OS scheduling) and never
        subtracts from it, so the fastest observed run is the closest
        estimator of the noise-free cost.
        """

        return min(values) if values else 0.0

    @staticmethod
    def geometric_mean(values: list[float]) -> float:
        """Return the geometric mean of ``values`` (``0`` for the empty list)."""

        if not values:
            return 0.0
        return statistics.geometric_mean(values)


@dataclass
class BenchmarkReport:
    """
    Define an accumulator of :class:`BenchmarkResult` records across suites.

    Parameters
    ----------
    results
        Initial list of results (defaults to empty).

    Attributes
    ----------
    -   :attr:`~BenchmarkReport.suites`
    -   :attr:`~BenchmarkReport.statistics`

    Methods
    -------
    -   :meth:`~BenchmarkReport.extend`
    -   :meth:`~BenchmarkReport.results_for`
    -   :meth:`~BenchmarkReport.rank_speedups`
    -   :meth:`~BenchmarkReport.log`
    -   :meth:`~BenchmarkReport.log_summary`
    -   :meth:`~BenchmarkReport.log_errors`
    -   :meth:`~BenchmarkReport.log_speedup_extremes`

    See Also
    --------
    -   :func:`write_report_json`
    -   :func:`write_report_csv`
    """

    results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def suites(self) -> list[str]:
        """Distinct suite names present in the report, in insertion order."""

        seen: dict[str, None] = {}
        for result in self.results:
            seen.setdefault(result.suite, None)
        return list(seen)

    @cached_property
    def statistics(self) -> dict[str, list[BackendStatistics]]:
        """Per-suite, per-backend aggregated statistics."""

        return {suite: self._statistics_for(suite) for suite in self.suites}

    def extend(self, results: list[BenchmarkResult]) -> None:
        """Extend the report with ``results`` and invalidate cached views."""

        self.results.extend(results)
        self.__dict__.pop("statistics", None)

    def results_for(self, suite: str) -> list[BenchmarkResult]:
        """Return all results belonging to ``suite``."""

        return [result for result in self.results if result.suite == suite]

    def _statistics_for(self, suite: str) -> list[BackendStatistics]:
        """Compute per-backend statistics for ``suite``."""

        suite_results = self.results_for(suite)

        numpy_durations = {
            result.label: result.duration
            for result in suite_results
            if result.backend == "numpy" and result.status == "SUCCEEDED"
        }

        backends = sorted(
            {result.backend for result in suite_results},
            key=DEFAULT_BENCHMARK_CONFIGURATION.backends.index,
        )

        per_backend = []
        for backend in backends:
            backend_results = [
                result for result in suite_results if result.backend == backend
            ]
            successful = [
                result
                for result in backend_results
                if result.status == "SUCCEEDED" and result.duration > 0
            ]

            speedups = [
                numpy_durations[result.label] / result.duration
                for result in successful
                if numpy_durations.get(result.label, 0) > 0
            ]

            per_backend.append(
                BackendStatistics(
                    backend=backend,
                    succeeded=sum(
                        1 for result in backend_results if result.status == "SUCCEEDED"
                    ),
                    skipped=sum(
                        1 for result in backend_results if result.status == "SKIPPED"
                    ),
                    failed=sum(
                        1
                        for result in backend_results
                        if result.status not in ("SUCCEEDED", "SKIPPED")
                    ),
                    duration=BackendStatistics.median(
                        [result.duration for result in successful]
                    ),
                    speedup=BackendStatistics.geometric_mean(speedups),
                )
            )
        return per_backend

    def rank_speedups(
        self,
        suite: str,
        backend: str,
    ) -> list[tuple[float, BenchmarkResult]]:
        """Return per-case ``(speedup, result)`` pairs sorted descending."""

        suite_results = self.results_for(suite)
        numpy_durations = {
            result.label: result.duration
            for result in suite_results
            if result.backend == "numpy" and result.status == "SUCCEEDED"
        }

        ranked = [
            (numpy_durations[result.label] / result.duration, result)
            for result in suite_results
            if result.backend == backend
            and result.status == "SUCCEEDED"
            and result.duration > 0
            and numpy_durations.get(result.label, 0) > 0
        ]
        return sorted(ranked, key=lambda x: -x[0])

    def log(self) -> None:
        """Log summary, errors, and speedup extremes for every suite."""

        for suite in self.suites:
            LOGGER.info("")
            LOGGER.info("BenchmarkSuite: %s", suite)
            self.log_summary(suite)
            self.log_errors(suite)
            self.log_speedup_extremes(suite)

    def log_summary(self, suite: str) -> None:
        """Log the per-backend summary table for ``suite``."""

        LOGGER.info(
            "%-12s %9s %7s %6s %10s %10s",
            "Backend",
            "Succeeded",
            "Skipped",
            "Failed",
            "Median",
            "Geo-mean",
        )
        LOGGER.info("-" * 58)

        for statistic in self.statistics[suite]:
            duration = (
                BenchmarkResult.format_duration(statistic.duration)
                if statistic.duration > 0
                else "-"
            )
            speedup = f"{statistic.speedup:.1f}x" if statistic.speedup > 0 else "-"
            LOGGER.info(
                "%-12s %9d %7d %6d %10s %10s",
                statistic.backend,
                statistic.succeeded,
                statistic.skipped,
                statistic.failed,
                duration,
                speedup,
            )

    def log_errors(self, suite: str) -> None:
        """Log a compact breakdown of failed cases within ``suite``."""

        errors = [
            result
            for result in self.results_for(suite)
            if result.status not in ("SUCCEEDED", "SKIPPED")
        ]
        if not errors:
            return

        LOGGER.info("Errors (%d):", len(errors))
        by_category = Counter((result.status, result.backend) for result in errors)
        for (category, backend), count in by_category.most_common():
            examples = [
                result
                for result in errors
                if result.status == category and result.backend == backend
            ][:3]
            labels = ", ".join(result.label for result in examples)
            LOGGER.info("  %s [%s] x%d: %s", category, backend, count, labels)

    def log_speedup_extremes(self, suite: str) -> None:
        """Log the top-10 and bottom-10 speedups per backend within ``suite``."""

        backends = sorted(
            {
                result.backend
                for result in self.results_for(suite)
                if result.backend != "numpy"
            },
            key=DEFAULT_BENCHMARK_CONFIGURATION.backends.index,
        )

        def _format(entries: list[tuple[float, BenchmarkResult]]) -> None:
            for speedup, result in entries:
                LOGGER.info(
                    "  %6.1fx  %s  (%s)",
                    speedup,
                    result.label,
                    BenchmarkResult.format_duration(result.duration),
                )

        for backend in backends:
            ranked = self.rank_speedups(suite, backend)
            if not ranked:
                continue

            LOGGER.info("Top 10 speedups (%s):", backend)
            _format(ranked[:10])

            LOGGER.info("Bottom 10 (%s):", backend)
            _format(ranked[-10:])


def write_report_json(report: BenchmarkReport, path: str) -> None:
    """Write a machine-readable JSON ``report`` to ``path`` (durations in seconds)."""

    document = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "suites": [
            {
                "name": suite,
                "summary": [
                    {
                        "backend": statistic.backend,
                        "succeeded": statistic.succeeded,
                        "skipped": statistic.skipped,
                        "failed": statistic.failed,
                        "duration": statistic.duration,
                        "speedup": statistic.speedup,
                    }
                    for statistic in report.statistics[suite]
                ],
                "results": [
                    {
                        "label": result.label,
                        "backend": result.backend,
                        "status": result.status,
                        "duration": result.duration,
                        "error": result.error,
                        "traceback": result.traceback,
                        "metadata": result.metadata,
                    }
                    for result in report.results_for(suite)
                ],
            }
            for suite in report.suites
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
    LOGGER.info("JSON report written to %s", path)


def write_report_csv(report: BenchmarkReport, path: str) -> None:
    """Write a flat CSV of all raw ``report`` results to ``path``."""

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["suite", "label", "backend", "status", "duration", "error"])
        for result in report.results:
            writer.writerow(
                [
                    result.suite,
                    result.label,
                    result.backend,
                    result.status,
                    result.duration,
                    result.error,
                ]
            )
    LOGGER.info("CSV report written to %s", path)


class BenchmarkRunner:
    """
    Define a runner orchestrating the execution of one or more suites.

    The runner owns shared execution concerns (backend dispatch, warmup /
    timed runs, memory release), while each :class:`BenchmarkSuite` defines
    what to benchmark; :meth:`run` iterates over them.

    Parameters
    ----------
    mode
        ``"quick"`` for correctness smoke tests with small inputs,
        ``"full"`` for HD timing.
    backends
        Backend identifiers to measure; defaults to
        :attr:`BenchmarkConfiguration.backends`. Unavailable backends are
        silently dropped via :meth:`available_backends`.
    warmup
        Untimed iterations before timing starts.
    runs
        Timed iterations; the best (minimum) is recorded per case.
    edges_filter
        Optional ``"src->tgt,src->tgt"`` filter consumed by
        :class:`BenchmarkSuite_ConversionGraph`.

    Attributes
    ----------
    -   :attr:`~BenchmarkRunner.resolved_backends`
    -   :attr:`~BenchmarkRunner.size`
    -   :attr:`~BenchmarkRunner.resolution`

    Methods
    -------
    -   :meth:`~BenchmarkRunner.__init__`
    -   :meth:`~BenchmarkRunner.available_backends`
    -   :meth:`~BenchmarkRunner.convert_to_backend`
    -   :meth:`~BenchmarkRunner.log_progress`
    -   :meth:`~BenchmarkRunner.run_case`
    -   :meth:`~BenchmarkRunner.run_cases`
    -   :meth:`~BenchmarkRunner.run`
    -   :meth:`~BenchmarkRunner.parse_suites`

    Examples
    --------
    >>> runner = BenchmarkRunner(mode="quick", backends=["numpy"])
    >>> report = runner.run([BenchmarkSuite_Difference(runner)])  # doctest: +SKIP
    >>> report.log()  # doctest: +SKIP
    """

    def __init__(
        self,
        mode: str = "quick",
        backends: list[str] | None = None,
        warmup: int = 1,
        runs: int = 1,
        edges_filter: str | None = None,
    ) -> None:
        self.mode = mode
        self.backends = backends
        self.warmup = warmup
        self.runs = runs
        self.edges_filter = edges_filter

    @cached_property
    def resolved_backends(self) -> list[str]:
        """Requested backends filtered to those available on this system."""

        return self.available_backends(
            self.backends or list(DEFAULT_BENCHMARK_CONFIGURATION.backends)
        )

    @cached_property
    def size(self) -> str:
        """Input-size tag (``"small"`` or ``"hd"``) matching :attr:`mode`."""

        return "small" if self.mode == "quick" else "hd"

    @cached_property
    def resolution(self) -> int:
        """Pixel count associated with :attr:`size`."""

        return DEFAULT_BENCHMARK_CONFIGURATION.resolution(self.size)

    # -- Backends ----------------------------------------------------------

    @staticmethod
    def available_backends(requested: list[str]) -> list[str]:
        """Return the subset of ``requested`` backends available on this system."""

        available = []
        for backend in requested:
            if (
                backend == "numpy"
                or (backend == "jax" and jax is not None)
                or (backend == "torch-cpu" and torch is not None)
            ):
                available.append(backend)
            elif backend == "torch-mps" and torch is not None:
                try:
                    if torch.backends.mps.is_available():
                        available.append(backend)
                except AttributeError:
                    continue
        return available

    @staticmethod
    def convert_to_backend(data: typing.Any, backend: str) -> typing.Any:
        """Promote a *NumPy* array to ``backend``, leaving other inputs unchanged."""

        if isinstance(data, tuple):
            return tuple(
                BenchmarkRunner.convert_to_backend(value, backend) for value in data
            )

        if is_dataclass(data) and not isinstance(data, type):
            promoted = {
                f.name: BenchmarkRunner.convert_to_backend(
                    getattr(data, f.name), backend
                )
                for f in fields(data)
            }
            return replace(data, **promoted)

        if not isinstance(data, np.ndarray) or backend == "numpy":
            return data

        if backend == "jax" and jax is not None and jnp is not None:
            jax.config.update("jax_enable_x64", True)
            data = jnp.array(data)
        elif backend == "torch-cpu" and torch is not None:
            data = torch.from_numpy(data.copy())
        elif backend == "torch-mps" and torch is not None:
            data = torch.from_numpy(data.astype(np.float32).copy()).to("mps")

        return data

    # -- Primitives --------------------------------------------------------

    def log_progress(self, done: int, total: int, result: BenchmarkResult) -> None:
        """Log a compact progress line for a finished case."""

        status = (
            BenchmarkResult.format_duration(result.duration)
            if result.status == "SUCCEEDED"
            else result.status
        )
        LOGGER.info(
            "[%d/%d] %s: %s  %s",
            done,
            total,
            result.backend,
            result.label,
            status,
        )

    def run_case(
        self,
        suite: str,
        backend: str,
        label: str,
        metadata: dict[str, str],
        operation: Callable[[], object],
    ) -> BenchmarkResult:
        """Run ``operation`` ``warmup`` + ``runs`` times, recording the best."""

        # ``torch-mps`` and ``jax`` both dispatch asynchronously and their
        # queues must be drained inside the timed region: without it the
        # measurement is the enqueue latency rather than the computation.
        # ``numpy`` and ``torch-cpu`` are synchronous from the host's
        # perspective.
        torch_mps = torch.mps if backend == "torch-mps" and torch is not None else None
        # *JAX* has no global barrier: the asynchronous computation is awaited
        # by blocking on the operation's own result.
        block = backend == "jax" and jax is not None

        def run_operation() -> None:
            """Run the operation, awaiting asynchronous backends."""

            with array_api_enable(backend != "numpy"):
                result = operation()

            if block:
                jax.block_until_ready(result)

        times: list[float] = []
        try:
            for _ in range(self.warmup):
                run_operation()
            if torch_mps is not None:
                torch_mps.synchronize()

            for _ in range(self.runs):
                if torch_mps is not None:
                    torch_mps.synchronize()
                start = time.perf_counter()
                run_operation()
                if torch_mps is not None:
                    torch_mps.synchronize()
                times.append(time.perf_counter() - start)
        # Bench isolation: a failing case (any exception other than
        # ``KeyboardInterrupt`` / ``SystemExit``) is recorded and the
        # runner continues with the next case. The broad catch is
        # intentional.
        except Exception as exception:  # noqa: BLE001
            return BenchmarkResult.failed(suite, backend, label, exception, metadata)

        gc.collect()
        if torch_mps is not None:
            torch_mps.empty_cache()

        return BenchmarkResult(
            suite=suite,
            backend=backend,
            status="SUCCEEDED",
            label=label,
            duration=BackendStatistics.best(times),
            metadata=metadata,
        )

    def run_cases(self, suite: BenchmarkSuite) -> list[BenchmarkResult]:
        """Iterate backends x suite cases, tracking progress and MPS dtype."""

        results: list[BenchmarkResult] = []
        total = suite.case_count * len(self.resolved_backends)
        done = 0

        for backend in self.resolved_backends:
            if backend == "torch-mps":
                set_default_float_dtype(np.float32)

            for label, metadata, operation in suite.cases(backend):
                done += 1
                if operation is None:
                    result = BenchmarkResult(
                        suite=suite.NAME,
                        backend=backend,
                        status="SKIPPED",
                        label=label,
                        metadata=metadata,
                    )
                else:
                    result = self.run_case(
                        suite.NAME, backend, label, metadata, operation
                    )
                results.append(result)
                self.log_progress(done, total, result)

            if backend == "torch-mps":
                set_default_float_dtype(np.float64)

        return results

    # -- Orchestration -----------------------------------------------------

    def run(self, suites: Iterable[BenchmarkSuite]) -> BenchmarkReport:
        """Run ``suites`` and return an accumulated :class:`BenchmarkReport`."""

        report = BenchmarkReport()
        for suite in suites:
            suite.log_header()
            report.extend(self.run_cases(suite))
        return report

    @classmethod
    def parse_suites(cls, value: str, runner: BenchmarkRunner) -> list[BenchmarkSuite]:
        """Resolve a comma-separated suite name string to a list of instances."""

        selected = []
        for name in value.split(","):
            stripped = name.strip()
            if stripped not in BENCHMARK_SUITES:
                message = (
                    f"Unknown suite: {stripped!r}. Available: {list(BENCHMARK_SUITES)}"
                )
                raise ValueError(message)
            selected.append(BENCHMARK_SUITES[stripped](runner))
        return selected


class BenchmarkSuite(ABC):
    """
    Abstract base class for a benchmark suite.

    A suite describes WHAT to benchmark: its name, how many cases it runs,
    and how to generate ``(label, metadata, operation)`` tuples per backend.
    Execution is delegated to :meth:`BenchmarkRunner.run_cases`.

    Parameters
    ----------
    runner
        Runner providing backend dispatch, warmup / run counts, and size.

    Attributes
    ----------
    -   :attr:`~BenchmarkSuite.NAME`
    -   :attr:`~BenchmarkSuite.case_count`

    Methods
    -------
    -   :meth:`~BenchmarkSuite.cases`
    """

    NAME: str = ""

    def __init__(self, runner: BenchmarkRunner) -> None:
        self._runner = runner
        self._rng = DEFAULT_BENCHMARK_CONFIGURATION.rng()

    @property
    @abstractmethod
    def case_count(self) -> int:
        """Total number of cases the suite will emit per backend."""

    @abstractmethod
    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """
        Yield ``(label, metadata, operation)`` tuples for ``backend``.

        An ``operation`` of ``None`` signals a preemptive skip: the runner
        records a :class:`BenchmarkResult` with ``status="SKIPPED"`` instead
        of executing anything.
        """

    def log_header(self) -> None:
        """Log a one-line header describing the suite before execution."""

        LOGGER.info("BenchmarkSuite: %s | Cases: %d", self.NAME, self.case_count)


# -----------------------------------------------------------------------------
# Concrete suites
# -----------------------------------------------------------------------------


class BenchmarkSuite_ConversionGraph(BenchmarkSuite):
    """
    Walk the array-path of the *Colour* automatic conversion graph.

    Methods backed by ``solve_CCT_Newton`` or
    ``scipy.spatial.distance.cdist`` are filtered out via
    :attr:`BenchmarkConfiguration.iterative_graph_cases` and benched
    separately in :class:`BenchmarkSuite_ConversionGraphIterative`. The
    geo-mean here therefore reflects *Array API*-dispatchable closed-form
    edges only.
    """

    NAME = "conversion_graph"

    @cached_property
    def edges(self) -> list[tuple[str, str]]:
        """``(source, target)`` edges honouring ``edges_filter`` / skip lists."""

        filter_set: frozenset[tuple[str, str]] | None = None
        if self._runner.edges_filter:
            parsed = set()
            for entry in self._runner.edges_filter.split(","):
                parts = entry.strip().split("->")
                if len(parts) == 2:
                    parsed.add((parts[0].strip(), parts[1].strip()))
            filter_set = frozenset(parsed)

        edges: list[tuple[str, str]] = []
        for source, target, _fn in CONVERSION_SPECIFICATIONS_DATA:
            if (
                source in DEFAULT_BENCHMARK_CONFIGURATION.skip_sources
                or target in DEFAULT_BENCHMARK_CONFIGURATION.skip_targets
            ):
                continue
            if filter_set is not None and (source, target) not in filter_set:
                continue
            edges.append((source, target))
        return edges

    @cached_property
    def edge_method_registries(self) -> dict[tuple[str, str], list[str]]:
        """Per-edge method lists, deduped from each registry."""

        edges: dict[tuple[str, str], list[str]] = {}
        for edge, registry in DEFAULT_BENCHMARK_CONFIGURATION.edge_method_registries:
            seen: set[int] = set()
            methods: list[str] = []
            for method, function in registry.items():
                key = id(function)
                if key not in seen:
                    seen.add(key)
                    methods.append(method)
            edges[edge] = methods
        return edges

    def _is_iterative_case(self, source: str, target: str, method: str | None) -> bool:
        """Return whether ``(source, target, method)`` is iterative-bound."""

        return (
            source,
            target,
            method,
        ) in DEFAULT_BENCHMARK_CONFIGURATION.iterative_graph_cases

    def _include_case(self, source: str, target: str, method: str | None) -> bool:
        """
        Per-class filter on whether a case belongs in this suite.

        Overridden by :class:`BenchmarkSuite_ConversionGraphIterative` to
        flip the filter direction.
        """

        return not self._is_iterative_case(source, target, method)

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        total = 0
        for source, target in self.edges:
            methods = self.edge_method_registries.get((source, target))
            if methods:
                total += sum(
                    1 for m in methods if self._include_case(source, target, m)
                )
            elif self._include_case(source, target, None):
                total += 1
        return total

    def log_header(self) -> None:
        """See :class:`BenchmarkSuite`."""

        LOGGER.info(
            "BenchmarkSuite: %s | Size: %s | Cases: %d",
            self.NAME,
            self._runner.size,
            self.case_count,
        )

    def generate_input(self, source: str, size: str) -> object:
        """Generate a representative input for a graph source node.

        A fresh seeded ``Generator`` is built per call rather than reusing
        ``self._rng`` so each ``(source, size)`` pair produces the same
        values on every backend pass, keeping cross-backend timings
        directly comparable. The trade-off is that distinct sources at
        the same shape see identical random draws; that is acceptable
        because each edge is timed in isolation.
        """

        n = DEFAULT_BENCHMARK_CONFIGURATION.resolution(size)
        rng = DEFAULT_BENCHMARK_CONFIGURATION.rng()

        if source in DEFAULT_BENCHMARK_CONFIGURATION.cam_sources:
            return dict(DEFAULT_BENCHMARK_CONFIGURATION.cam_specifications)[source](
                J=rng.random(n) * 100,
                M=rng.random(n),
                h=rng.random(n) * 360,
            )

        if source == "CMYK":
            return rng.random((n, 4))

        if source.startswith("CCT") or source.endswith((" xy", " uv")):
            return rng.random((n, 2))

        return rng.random((n, 3))

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        for source, target in self.edges:
            base_metadata = {"source": source, "target": target}

            if (
                backend == "torch-mps"
                and (source, target) in DEFAULT_BENCHMARK_CONFIGURATION.skip_mps_edges
            ):
                yield f"{source} -> {target}", base_metadata, None
                continue

            edge_size = (
                "reduced"
                if self._runner.size == "hd"
                and (
                    target in DEFAULT_BENCHMARK_CONFIGURATION.reduced_size_targets
                    or (source, target)
                    in DEFAULT_BENCHMARK_CONFIGURATION.reduced_size_edges
                )
                else self._runner.size
            )
            data = self.generate_input(source, edge_size)
            if backend != "numpy":
                data = BenchmarkRunner.convert_to_backend(data, backend)

            methods = self.edge_method_registries.get((source, target))
            if not methods:
                if not self._include_case(source, target, None):
                    continue
                yield (
                    f"{source} -> {target}",
                    base_metadata,
                    lambda x=data, s=source, t=target: convert(x, s, t),
                )
                continue

            for method in methods:
                if not self._include_case(source, target, method):
                    continue
                yield (
                    f"{source} -> {target} ({method})",
                    {**base_metadata, "method": method},
                    lambda x=data, s=source, t=target, m=method: convert(
                        x, s, t, method=m
                    ),
                )


class BenchmarkSuite_ConversionGraphIterative(BenchmarkSuite_ConversionGraph):
    """Iterative-method counterpart to :class:`BenchmarkSuite_ConversionGraph`.

    Walks the same graph but yields only the cases flagged in
    :attr:`BenchmarkConfiguration.iterative_graph_cases`: methods backed
    by ``solve_CCT_Newton`` (``Planck 1900``, ``Krystek 1985``) or
    ``scipy.spatial.distance.cdist`` (``Colorimetric Purity``,
    ``Excitation Purity``). The geo-mean here measures Python-loop
    overhead, not *Array API* dispatch; a faster backend cannot
    accelerate these methods without an algorithmic restructure.
    """

    NAME = "conversion_graph_iterative"

    def _include_case(self, source: str, target: str, method: str | None) -> bool:
        """Invert the parent filter: yield only iterative-bound cases."""

        return self._is_iterative_case(source, target, method)


class BenchmarkSuite_Difference(BenchmarkSuite):
    """Benchmark every Delta E function on an ``(N, 3)`` pair."""

    NAME = "difference"

    def __init__(self, runner: BenchmarkRunner) -> None:
        super().__init__(runner)
        self._a_np = self._rng.random((runner.resolution, 3)) * 100
        self._b_np = self._rng.random((runner.resolution, 3)) * 100

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        return len(DELTA_E_METHODS)

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        a = BenchmarkRunner.convert_to_backend(self._a_np, backend)
        b = BenchmarkRunner.convert_to_backend(self._b_np, backend)
        for method, function in DELTA_E_METHODS.items():
            yield (
                method,
                {"method": method},
                lambda function=function, x=a, y=b: function(x, y),
            )


class BenchmarkSuite_IntegrationArray(BenchmarkSuite):
    """
    Benchmark vectorised spectral integration via the array-path of
    ``sd_to_XYZ`` / ``msds_to_XYZ`` with raw ``ArrayLike`` inputs and
    ``method="Integration"``.

    Both cases dispatch through the *Array API* and follow the input
    backend; ``msds_to_XYZ array`` exercises an HD multi-spectral image
    at 1 nm resolution and is the integration suite's headline
    backend-acceleration case.
    """

    NAME = "integration_array"

    def __init__(self, runner: BenchmarkRunner) -> None:
        super().__init__(runner)
        self._shape = SpectralShape(380, 780, 1)
        self._cmfs = (
            MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(self._shape)
        )
        self._illuminant = SDS_ILLUMINANTS["D65"].copy().align(self._shape)
        n_wavelengths = self._shape.wavelengths.shape[0]
        self._sd_values_np = self._rng.random(n_wavelengths)
        self._msds_values_np = self._rng.random((runner.resolution, n_wavelengths))

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        return 2

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        sd_array = BenchmarkRunner.convert_to_backend(self._sd_values_np, backend)
        msds_array = BenchmarkRunner.convert_to_backend(self._msds_values_np, backend)

        yield (
            "sd_to_XYZ array (Integration)",
            {"operation": "sd_to_XYZ array", "method": "Integration"},
            lambda x=sd_array: sd_to_XYZ(
                x,
                cmfs=self._cmfs,
                illuminant=self._illuminant,
                method="Integration",
                shape=self._shape,
            ),
        )
        yield (
            "msds_to_XYZ array (Integration)",
            {"operation": "msds_to_XYZ array", "method": "Integration"},
            lambda x=msds_array: msds_to_XYZ(
                x,
                cmfs=self._cmfs,
                illuminant=self._illuminant,
                method="Integration",
                shape=self._shape,
            ),
        )


class BenchmarkSuite_IntegrationObject(BenchmarkSuite):
    """
    Benchmark per-:class:`SpectralDistribution` integration via the
    object-path of ``sd_to_XYZ`` / ``msds_to_XYZ`` across every registered
    method.

    The :class:`SpectralDistribution` and
    :class:`MultiSpectralDistributions` instances are constructed per
    backend with backend-typed values, so the integration helpers
    dispatch through the *Array API* and follow the input backend rather
    than collapsing to a numpy-only baseline.
    """

    NAME = "integration_object"

    def __init__(self, runner: BenchmarkRunner) -> None:
        super().__init__(runner)
        self._shape = SpectralShape(380, 780, 1)
        self._cmfs = (
            MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(self._shape)
        )
        self._illuminant = SDS_ILLUMINANTS["D65"].copy().align(self._shape)
        n_wavelengths = self._shape.wavelengths.shape[0]
        # Small ``MSDS`` matching a colour-checker chart in ``quick``
        # mode and a small spectral image in ``full`` mode.
        n_msds = 24 if runner.mode == "quick" else 1024
        self._sd_values_np = self._rng.random(n_wavelengths)
        self._msds_values_np = self._rng.random((n_wavelengths, n_msds))

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        return len(SD_TO_XYZ_METHODS) + len(MSDS_TO_XYZ_METHODS)

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        sd_values = BenchmarkRunner.convert_to_backend(self._sd_values_np, backend)
        msds_values = BenchmarkRunner.convert_to_backend(self._msds_values_np, backend)
        with array_api_enable(backend != "numpy"):
            sd = SpectralDistribution(sd_values, self._shape.wavelengths)
            msds = MultiSpectralDistributions(msds_values, self._shape.wavelengths)

        for method in SD_TO_XYZ_METHODS:
            yield (
                f"sd_to_XYZ ({method})",
                {"operation": "sd_to_XYZ", "method": method},
                lambda method=method, x=sd: sd_to_XYZ(
                    x,
                    cmfs=self._cmfs,
                    illuminant=self._illuminant,
                    method=method,
                ),
            )
        for method in MSDS_TO_XYZ_METHODS:
            yield (
                f"msds_to_XYZ ({method})",
                {"operation": "msds_to_XYZ", "method": method},
                lambda method=method, x=msds: msds_to_XYZ(
                    x,
                    cmfs=self._cmfs,
                    illuminant=self._illuminant,
                    method=method,
                ),
            )


class BenchmarkSuite_TransferFunction(BenchmarkSuite):
    """Benchmark every CCTF encoding and decoding across backends."""

    NAME = "transfer_function"

    def __init__(self, runner: BenchmarkRunner) -> None:
        super().__init__(runner)
        self._a_np = self._rng.random(runner.resolution)
        self._directions: list[tuple[str, Mapping[str, Callable[..., object]]]] = [
            ("encoding", CCTF_ENCODINGS),
            ("decoding", CCTF_DECODINGS),
        ]

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        return sum(len(mapping) for _, mapping in self._directions)

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        a = BenchmarkRunner.convert_to_backend(self._a_np, backend)
        for direction, mapping in self._directions:
            for function_name, function in mapping.items():
                yield (
                    f"{direction} {function_name}",
                    {"direction": direction, "function": function_name},
                    lambda function=function, x=a: function(x),
                )


class BenchmarkSuite_Adaptation(BenchmarkSuite):
    """Benchmark direct chromatic adaptation (matrix build + apply)."""

    NAME = "adaptation"

    _EXTRA_KWARGS: typing.ClassVar[dict[str, dict[str, float]]] = {
        "CIE 1994": {"Y_o": 0.2, "E_o1": 1000.0, "E_o2": 1000.0},
        "CMCCAT2000": {"L_A1": 200.0, "L_A2": 200.0},
        "Fairchild 1990": {"Y_n": 200.0},
        "Li 2025": {"L_A": 200.0, "F_surround": 1.0},
    }

    def __init__(self, runner: BenchmarkRunner) -> None:
        super().__init__(runner)
        self._XYZ_np = self._rng.random((runner.resolution, 3))
        self._XYZ_w_np = np.array([0.95045593, 1.0, 1.08905775])
        self._XYZ_wr_np = np.array([0.96429568, 1.0, 0.82510460])

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        return len(CHROMATIC_ADAPTATION_METHODS)

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        XYZ = BenchmarkRunner.convert_to_backend(self._XYZ_np, backend)
        XYZ_w = BenchmarkRunner.convert_to_backend(self._XYZ_w_np, backend)
        XYZ_wr = BenchmarkRunner.convert_to_backend(self._XYZ_wr_np, backend)
        for method in CHROMATIC_ADAPTATION_METHODS:
            yield (
                f"CAT {method}",
                {"method": method},
                partial(
                    chromatic_adaptation,
                    XYZ,
                    XYZ_w,
                    XYZ_wr,
                    method=method,
                    **self._EXTRA_KWARGS.get(method, {}),
                ),
            )


class BenchmarkSuite_Characterisation(BenchmarkSuite):
    """Benchmark colour correction matrix methods (build + apply)."""

    NAME = "characterisation"

    def __init__(self, runner: BenchmarkRunner) -> None:
        super().__init__(runner)
        self._M_T_np = self._rng.random((24, 3))
        self._M_R_np = self._rng.random((24, 3))
        self._RGB_np = self._rng.random((runner.resolution, 3))

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        return len(MATRIX_COLOUR_CORRECTION_METHODS)

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        M_T = BenchmarkRunner.convert_to_backend(self._M_T_np, backend)
        M_R = BenchmarkRunner.convert_to_backend(self._M_R_np, backend)
        RGB = BenchmarkRunner.convert_to_backend(self._RGB_np, backend)
        for method in MATRIX_COLOUR_CORRECTION_METHODS:
            yield (
                f"CCM {method}",
                {"method": method},
                lambda method=method, M_T=M_T, M_R=M_R, RGB=RGB: (
                    apply_matrix_colour_correction(
                        RGB,
                        matrix_colour_correction(M_T, M_R, method=method),
                        method=method,
                    )
                ),
            )


class BenchmarkSuite_RecoveryArray(BenchmarkSuite):
    """
    Benchmark vectorisable spectral recovery methods on an ``(N, 3)``
    *RGB* batch sized by :attr:`BenchmarkRunner.resolution`.

    Covers *Smits 1999*, *Gaussian*, *Mallett 2019* (basis-expansion
    methods) and the *Jakob 2019* LUT runtime path
    (:class:`LUT3D_Jakob2019.RGB_to_coefficients`), which all natively
    support batched array input. The *Jakob 2019* offline solver
    (:func:`XYZ_to_sd_Jakob2019`) is benched separately in
    :class:`BenchmarkSuite_RecoveryObject`.
    """

    NAME = "recovery_array"

    def __init__(self, runner: BenchmarkRunner) -> None:
        super().__init__(runner)
        self._RGB_batch_np = self._rng.random((runner.resolution, 3)) * 0.5
        self._basis_functions_mallett_np = np.asarray(
            MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019.values
        )

        # ``LUT3D_Jakob2019.generate`` is offline (one optimisation per
        # grid cell); a small grid at 10 nm step keeps bench-time setup
        # interactive. Runtime ``RGB_to_coefficients`` cost is
        # independent of grid size for HD batches.
        shape = SpectralShape(
            SPECTRAL_SHAPE_JAKOB2019.start, SPECTRAL_SHAPE_JAKOB2019.end, 10
        )
        cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(shape)
        illuminant = SDS_ILLUMINANTS["D65"].copy().align(shape)
        self._jakob2019_lut = LUT3D_Jakob2019()
        self._jakob2019_lut.generate(RGB_COLOURSPACE_sRGB, cmfs, illuminant, size=5)

    @cached_property
    def _gaussian_basis(self) -> MultiSpectralDistributions:
        """Return a 10nm *Gaussian* recovery basis for ``RGB_to_msds_Gaussian``.

        The default ``MSDS_GAUSSIAN_BASIS`` ships at 1nm
        (``SPECTRAL_SHAPE_DEFAULT``, 421 wavelengths) which produces a
        ~7 GB float64 output at HD resolution. The basis spectra are
        smooth clamped Gaussians with no features narrower than ~30nm,
        so 10nm sampling is colorimetrically equivalent and an order
        of magnitude cheaper in memory bandwidth; representative of
        how callers actually use the recovery for image reproduction.
        Built lazily so unrelated suite invocations don't pay for it.
        """

        return generate_gaussian_basis(SpectralShape(360, 780, 10))

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        return 4

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        RGB_batch = BenchmarkRunner.convert_to_backend(self._RGB_batch_np, backend)
        basis_mallett = BenchmarkRunner.convert_to_backend(
            self._basis_functions_mallett_np, backend
        )

        yield (
            "RGB_to_msds_Smits1999",
            {"method": "Smits 1999"},
            lambda x=RGB_batch: RGB_to_msds_Smits1999(x, as_array=True),
        )
        # ``RGB_to_msds_Gaussian`` is a thin wrapper over
        # ``RGB_to_msds_Smits1999`` that pins ``MSDS_GAUSSIAN_BASIS`` at
        # the 1nm module default. We call the underlying function
        # directly with a 10nm basis so the bench measures algorithm
        # cost rather than the 1nm memory-bandwidth wall.
        yield (
            "RGB_to_msds_Gaussian (10nm basis)",
            {"method": "Gaussian"},
            lambda x=RGB_batch, b=self._gaussian_basis: RGB_to_msds_Smits1999(
                x, b, as_array=True
            ),
        )
        yield (
            "RGB_to_msds_Mallett2019",
            {"method": "Mallett 2019"},
            lambda x=RGB_batch, b=basis_mallett: x @ b.T,
        )
        yield (
            "LUT3D_Jakob2019.RGB_to_coefficients",
            {"method": "Jakob 2019 LUT"},
            lambda lut=self._jakob2019_lut, x=RGB_batch: lut.RGB_to_coefficients(x),
        )


class BenchmarkSuite_RecoveryObject(BenchmarkSuite):
    """
    Benchmark per-pixel spectral recovery solvers on a single
    ``(3,)`` *CIE XYZ* input.

    Covers *Jakob 2019*, *Meng 2015*, and *Otsu 2018*: per-pixel
    optimisers / decision-tree dispatch that do not vectorise across
    samples.
    """

    NAME = "recovery_object"

    _PER_PIXEL_METHODS: typing.ClassVar[tuple[str, ...]] = (
        "Jakob 2019",
        "Meng 2015",
        "Otsu 2018",
    )

    def __init__(self, runner: BenchmarkRunner) -> None:
        super().__init__(runner)
        self._XYZ_np = np.array([0.21, 0.18, 0.08])

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        return len(self._PER_PIXEL_METHODS)

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        XYZ = BenchmarkRunner.convert_to_backend(self._XYZ_np, backend)
        for method in self._PER_PIXEL_METHODS:
            yield (
                f"XYZ_to_sd ({method})",
                {"method": method},
                lambda method=method, x=XYZ: XYZ_to_sd(x, method=method),
            )


class BenchmarkSuite_QualityArray(BenchmarkSuite):
    """
    Benchmark light-source quality indices on batched
    :class:`MultiSpectralDistributions` inputs.

    Covers the *CRI*, *CFI* (*CIE 2017* and *ANSI/IES TM-30-18*) and
    *CQS* batch paths. ``spectral_similarity_index`` accepts a single
    :class:`SpectralDistribution` only and is not exercised here. The
    batch path computes the *Planckian* / *CIE D Series* references via
    the array kernels (``planck_law``, ``CIE_illuminant_D_series``)
    rather than building an :class:`MSDS` for each batch entry, so this
    suite captures the win from skipping that construction. The
    :class:`MultiSpectralDistributions` values are promoted to the
    requested backend in :meth:`cases` so that *Array API* dispatch
    propagates through the entire pipeline.
    """

    NAME = "quality_array"

    def __init__(self, runner: BenchmarkRunner) -> None:
        super().__init__(runner)
        n = 24 if runner.mode == "quick" else 1024
        self._sd_template = SDS_LIGHT_SOURCES["Cool White FL"]
        # Synthetic batch: ``sd_template`` replicated and perturbed by
        # a small uniform jitter so the *CCT* solver stays well-posed
        # while the test exercises the full vectorised pipeline.
        self._values_np = self._rng.random(
            (self._sd_template.values.shape[0], n)
        ) * 0.05 + (self._sd_template.values[:, None] * 0.95)
        self._labels = [f"{self._sd_template.name} #{i}" for i in range(n)]

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        return 4

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        # The :class:`MultiSpectralDistributions` constructor calls
        # ``as_float_array`` on the input values which falls back to
        # ``np.asarray`` outside the *Array API* dispatch context;
        # building the *MSDS* inside ``array_api_enable`` keeps backend
        # tensors (notably *PyTorch MPS*) without forcing a host
        # round-trip.
        values = BenchmarkRunner.convert_to_backend(self._values_np, backend)
        with array_api_enable(backend != "numpy"):
            msds_test = MultiSpectralDistributions(
                values, self._sd_template.wavelengths, labels=self._labels
            )

        yield (
            "colour_rendering_index",
            {"function": "colour_rendering_index"},
            lambda: colour_rendering_index(msds_test),
        )
        yield (
            "colour_fidelity_index_CIE2017",
            {"function": "colour_fidelity_index_CIE2017"},
            lambda: colour_fidelity_index(msds_test),
        )
        yield (
            "colour_fidelity_index_ANSIIESTM3018",
            {"function": "colour_fidelity_index_ANSIIESTM3018"},
            lambda: colour_fidelity_index(
                msds_test,
                method="ANSI/IES TM-30-18",
            ),
        )
        yield (
            "colour_quality_scale",
            {"function": "colour_quality_scale"},
            lambda: colour_quality_scale(msds_test),
        )


class BenchmarkSuite_QualityObject(BenchmarkSuite):
    """
    Benchmark light-source quality indices on single
    :class:`SpectralDistribution` inputs.

    Covers *CRI*, *CFI* (*CIE 2017* and *ANSI/IES TM-30-18*), *CQS* and
    *SSI*. The :class:`SpectralDistribution` instances are rebuilt per
    backend with backend-typed values so the underlying integration
    paths dispatch through the *Array API* rather than running the same
    numpy compute under every backend.
    """

    NAME = "quality_object"

    def __init__(self, runner: BenchmarkRunner) -> None:
        super().__init__(runner)
        self._sd_test_template = SDS_LIGHT_SOURCES["Cool White FL"]
        self._sd_reference_template = SDS_LIGHT_SOURCES["Daylight FL"]

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        return 5

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        test_values = BenchmarkRunner.convert_to_backend(
            self._sd_test_template.values, backend
        )
        reference_values = BenchmarkRunner.convert_to_backend(
            self._sd_reference_template.values, backend
        )
        with array_api_enable(backend != "numpy"):
            sd_test = SpectralDistribution(
                test_values,
                self._sd_test_template.wavelengths,
                name=self._sd_test_template.name,
            )
            sd_reference = SpectralDistribution(
                reference_values,
                self._sd_reference_template.wavelengths,
                name=self._sd_reference_template.name,
            )

        yield (
            "colour_rendering_index",
            {"function": "colour_rendering_index"},
            lambda: colour_rendering_index(sd_test),
        )
        yield (
            "colour_fidelity_index_CIE2017",
            {"function": "colour_fidelity_index_CIE2017"},
            lambda: colour_fidelity_index(sd_test),
        )
        yield (
            "colour_fidelity_index_ANSIIESTM3018",
            {"function": "colour_fidelity_index_ANSIIESTM3018"},
            lambda: colour_fidelity_index(sd_test, method="ANSI/IES TM-30-18"),
        )
        yield (
            "colour_quality_scale",
            {"function": "colour_quality_scale"},
            lambda: colour_quality_scale(sd_test),
        )
        yield (
            "spectral_similarity_index",
            {"function": "spectral_similarity_index"},
            lambda: spectral_similarity_index(sd_test, sd_reference),
        )


class BenchmarkSuite_Volume(BenchmarkSuite):
    """
    Benchmark the *Array API*-dispatchable gamut volume cases:
    ``RGB_colourspace_volume_MonteCarlo`` and ``RGB_colourspace_limits``.

    The ``scipy.spatial.Delaunay``-backed containment checks live in
    :class:`BenchmarkSuite_VolumeIterative` so the geo-mean here is not
    polluted by host-only ``Delaunay`` triangulation.
    """

    NAME = "volume"

    def __init__(self, runner: BenchmarkRunner) -> None:
        super().__init__(runner)
        self._colourspace = RGB_COLOURSPACES["sRGB"]
        self._samples = 10_000 if runner.mode == "quick" else 250_000

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        return 2

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        del backend
        yield (
            "RGB_colourspace_volume_MonteCarlo",
            {"function": "RGB_colourspace_volume_MonteCarlo"},
            lambda: RGB_colourspace_volume_MonteCarlo(
                self._colourspace, samples=self._samples
            ),
        )
        yield (
            "RGB_colourspace_limits",
            {"function": "RGB_colourspace_limits"},
            lambda: RGB_colourspace_limits(self._colourspace),
        )


class BenchmarkSuite_VolumeIterative(BenchmarkSuite):
    """``scipy.spatial.Delaunay``-backed gamut containment checks.

    Counterpart to :class:`BenchmarkSuite_Volume`. The two cases here
    (``is_within_visible_spectrum``, ``is_within_macadam_limits``)
    forcibly round-trip through host memory for the *Delaunay*
    triangulation and ``find_simplex`` query, so they don't benefit
    from *Array API* dispatch.
    """

    NAME = "volume_iterative"

    def __init__(self, runner: BenchmarkRunner) -> None:
        super().__init__(runner)
        n = DEFAULT_BENCHMARK_CONFIGURATION.resolution(
            "small" if runner.mode == "quick" else "reduced"
        )
        self._XYZ_np = self._rng.random((n, 3))
        self._xyY_np = self._rng.random((n, 3))

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        return 2

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        XYZ = BenchmarkRunner.convert_to_backend(self._XYZ_np, backend)
        xyY = BenchmarkRunner.convert_to_backend(self._xyY_np, backend)

        yield (
            "is_within_visible_spectrum",
            {"function": "is_within_visible_spectrum"},
            lambda XYZ=XYZ: is_within_visible_spectrum(XYZ),
        )
        yield (
            "is_within_macadam_limits",
            {"function": "is_within_macadam_limits"},
            lambda xyY=xyY: is_within_macadam_limits(xyY),
        )


class BenchmarkSuite_Phenomena(BenchmarkSuite):
    """
    Benchmark sky models and *Rayleigh* scattering.

    ``rayleigh_optical_depth`` and the three *CIE 2003* sky distributions
    take array inputs (``wavelengths``, ``zenith``, ``azimuth``) and
    dispatch through the *Array API*; ``sd_rayleigh_scattering`` returns a
    :class:`SpectralDistribution` and stays numpy-bound.
    """

    NAME = "phenomena"

    def __init__(self, runner: BenchmarkRunner) -> None:
        super().__init__(runner)
        self._wavelengths_np = SpectralShape(360, 780, 1).wavelengths
        self._zenith_np = self._rng.random(runner.resolution) * (np.pi / 2)
        self._azimuth_np = self._rng.random(runner.resolution) * (2 * np.pi)
        self._z_sun = np.pi / 4
        self._a_sun = np.pi

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        return 5

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        wavelengths = BenchmarkRunner.convert_to_backend(self._wavelengths_np, backend)
        zenith = BenchmarkRunner.convert_to_backend(self._zenith_np, backend)
        azimuth = BenchmarkRunner.convert_to_backend(self._azimuth_np, backend)

        yield (
            "rayleigh_optical_depth",
            {"function": "rayleigh_optical_depth"},
            lambda w=wavelengths: rayleigh_optical_depth(w),
        )
        yield (
            "sd_rayleigh_scattering",
            {"function": "sd_rayleigh_scattering"},
            sd_rayleigh_scattering,
        )
        yield (
            "sky_luminance_distribution_CIE2003",
            {"function": "sky_luminance_distribution_CIE2003"},
            lambda z=zenith, a=azimuth, zs=self._z_sun, az=self._a_sun: (
                sky_luminance_distribution_CIE2003(1, z, a, zs, az)
            ),
        )
        yield (
            "sky_luminance_distribution_overcast_CIE2003",
            {"function": "sky_luminance_distribution_overcast_CIE2003"},
            lambda z=zenith: sky_luminance_distribution_overcast_CIE2003(z),
        )
        yield (
            "sky_scattering_indicatrix_CIE2003",
            {"function": "sky_scattering_indicatrix_CIE2003"},
            lambda z=zenith, a=azimuth, zs=self._z_sun, az=self._a_sun: (
                sky_scattering_indicatrix_CIE2003(z, a, zs, az)
            ),
        )


class BenchmarkSuite_TemperatureArray(BenchmarkSuite):
    """
    Benchmark closed-form *CIE xy* <-> correlated colour temperature
    conversions.

    *CIE UCS uv* and ``CCT`` and the colorimetry indices (whiteness,
    yellowness, lightness, luminance, Munsell value) are exercised through
    :class:`BenchmarkSuite_ConversionGraph` per-method fan-out and are
    deliberately not duplicated here.
    """

    NAME = "temperature_array"

    _XY_TO_CCT: typing.ClassVar[tuple[str, ...]] = ("Hernandez 1999", "McCamy 1992")
    _CCT_TO_XY: typing.ClassVar[tuple[str, ...]] = (
        "CIE Illuminant D Series",
        "Kang 2002",
    )

    def __init__(self, runner: BenchmarkRunner) -> None:
        super().__init__(runner)
        self._xy_np = self._rng.random((runner.resolution, 2)) * 0.4 + 0.2
        self._CCT_np = self._rng.random(runner.resolution) * 9000 + 1000

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        return len(self._XY_TO_CCT) + len(self._CCT_TO_XY)

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        xy = BenchmarkRunner.convert_to_backend(self._xy_np, backend)
        CCT = BenchmarkRunner.convert_to_backend(self._CCT_np, backend)

        for method in self._XY_TO_CCT:
            yield (
                f"xy_to_CCT ({method})",
                {"direction": "xy_to_CCT", "method": method},
                lambda method=method, c=xy: xy_to_CCT(c, method=method),
            )
        for method in self._CCT_TO_XY:
            yield (
                f"CCT_to_xy ({method})",
                {"direction": "CCT_to_xy", "method": method},
                lambda method=method, t=CCT: CCT_to_xy(t, method=method),
            )


class BenchmarkSuite_TemperatureIterative(BenchmarkSuite):
    """
    Benchmark iterative *CIE xy* <-> correlated colour temperature
    conversions backed by per-sample *Newton* solvers.

    The methods batch through :func:`colour.temperature.solve_CCT_Newton`
    and :func:`colour.temperature.solve_xy_Newton`; each sample converges
    independently so the suite is benched at ``reduced`` resolution to
    keep the iteration cost tractable in ``hd`` mode.
    """

    NAME = "temperature_iterative"

    _XY_TO_CCT: typing.ClassVar[tuple[str, ...]] = (
        "CIE Illuminant D Series",
        "Kang 2002",
    )
    _CCT_TO_XY: typing.ClassVar[tuple[str, ...]] = ("Hernandez 1999", "McCamy 1992")

    def __init__(self, runner: BenchmarkRunner) -> None:
        super().__init__(runner)
        n = (
            DEFAULT_BENCHMARK_CONFIGURATION.resolution("reduced")
            if runner.size == "hd"
            else runner.resolution
        )
        self._xy_np = self._rng.random((n, 2)) * 0.4 + 0.2
        self._CCT_np = self._rng.random(n) * 9000 + 1000

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        return len(self._XY_TO_CCT) + len(self._CCT_TO_XY)

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        xy = BenchmarkRunner.convert_to_backend(self._xy_np, backend)
        CCT = BenchmarkRunner.convert_to_backend(self._CCT_np, backend)

        for method in self._XY_TO_CCT:
            yield (
                f"xy_to_CCT ({method})",
                {"direction": "xy_to_CCT", "method": method},
                lambda method=method, c=xy: xy_to_CCT(c, method=method),
            )
        for method in self._CCT_TO_XY:
            yield (
                f"CCT_to_xy ({method})",
                {"direction": "CCT_to_xy", "method": method},
                lambda method=method, t=CCT: CCT_to_xy(t, method=method),
            )


class BenchmarkSuite_Blindness(BenchmarkSuite):
    """
    Benchmark *Machado (2009)* colour vision deficiency simulation.

    Each case builds the deficiency matrix and applies it to an HD *RGB*
    image, mirroring a typical CVD simulation pipeline.
    """

    NAME = "blindness"

    _DEFICIENCIES: typing.ClassVar[tuple[str, ...]] = (
        "Protanomaly",
        "Deuteranomaly",
        "Tritanomaly",
    )
    _SEVERITY: typing.ClassVar[float] = 0.5

    def __init__(self, runner: BenchmarkRunner) -> None:
        super().__init__(runner)
        self._RGB_np = self._rng.random((runner.resolution, 3))

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        return len(self._DEFICIENCIES)

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        RGB = BenchmarkRunner.convert_to_backend(self._RGB_np, backend)
        for deficiency in self._DEFICIENCIES:

            def operation(
                d: str = deficiency,
                x: typing.Any = RGB,
                s: float = self._SEVERITY,
            ) -> object:
                matrix = matrix_cvd_Machado2009(d, s)
                matrix = BenchmarkRunner.convert_to_backend(matrix, backend)
                return vecmul(matrix, x)

            yield (
                f"matrix_cvd_Machado2009 ({deficiency})",
                {"deficiency": deficiency},
                operation,
            )


class BenchmarkSuite_Contrast(BenchmarkSuite):
    """Benchmark *Barten (1999)* contrast sensitivity function."""

    NAME = "contrast"

    def __init__(self, runner: BenchmarkRunner) -> None:
        super().__init__(runner)
        self._frequencies_np = np.linspace(1.0, 30.0, runner.resolution)

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        return 1

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        frequencies = BenchmarkRunner.convert_to_backend(self._frequencies_np, backend)
        yield (
            "contrast_sensitivity_function (Barten 1999)",
            {"method": "Barten 1999"},
            lambda u=frequencies: contrast_sensitivity_function(
                u=u, method="Barten 1999"
            ),
        )


class BenchmarkSuite_GeneratorsArray(BenchmarkSuite):
    """
    Benchmark vectorised spectral generators with raw ``ArrayLike`` inputs
    and outputs.

    ``planck_law`` and ``rayleigh_jeans_law`` operate purely on
    wavelength and temperature arrays, broadcast to a
    ``(n_wavelengths, n_temperatures)`` output and dispatch through the
    *Array API*. ``CIE_illuminant_D_series`` evaluates the *CIE D Series*
    basis at a batch of *CIE xy* chromaticities, returning shape
    ``(n_wavelengths, N)``. The temperature / chromaticity batches are
    taken at ``"reduced"`` resolution in ``hd`` mode so broadcasted
    outputs stay inside a typical GPU memory budget.
    """

    NAME = "generators_array"

    def __init__(self, runner: BenchmarkRunner) -> None:
        super().__init__(runner)
        self._shape = SpectralShape(380, 780, 1)
        self._wavelengths_np = self._shape.wavelengths
        n = (
            DEFAULT_BENCHMARK_CONFIGURATION.resolution("reduced")
            if runner.size == "hd"
            else runner.resolution
        )
        self._temperatures_np = np.linspace(2000, 10000, n)
        self._xy_np = self._rng.random((n, 2)) * 0.1 + (0.31, 0.33)

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        return 3

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        wavelengths = BenchmarkRunner.convert_to_backend(self._wavelengths_np, backend)
        temperatures = BenchmarkRunner.convert_to_backend(
            self._temperatures_np, backend
        )
        xy = BenchmarkRunner.convert_to_backend(self._xy_np, backend)

        yield (
            "planck_law",
            {"function": "planck_law"},
            lambda w=wavelengths, t=temperatures: planck_law(w * 1e-9, t),
        )
        yield (
            "rayleigh_jeans_law",
            {"function": "rayleigh_jeans_law"},
            lambda w=wavelengths, t=temperatures: rayleigh_jeans_law(w * 1e-9, t),
        )
        yield (
            "CIE_illuminant_D_series",
            {"function": "CIE_illuminant_D_series"},
            lambda x=xy: CIE_illuminant_D_series(x, shape=self._shape),
        )


class BenchmarkSuite_GeneratorsObject(BenchmarkSuite):
    """
    Benchmark per-:class:`SpectralDistribution` /
    :class:`MultiSpectralDistributions` spectral generators.

    The ``sd_*`` cases take scalar inputs and construct a single
    :class:`SpectralDistribution` per call: the path is numpy-bound and
    backend-agnostic. The ``msds_*`` cases take array inputs that are
    promoted to the requested backend in :meth:`cases` so the underlying
    array kernels (``planck_law``, ``rayleigh_jeans_law``,
    ``CIE_illuminant_D_series``) dispatch through the *Array API* before
    the result is wrapped into a :class:`MultiSpectralDistributions`.
    """

    NAME = "generators_object"

    def __init__(self, runner: BenchmarkRunner) -> None:
        super().__init__(runner)
        self._shape = SpectralShape(380, 780, 1)
        # Multi-spectral fixtures match a colour-checker chart in
        # ``quick`` mode and a small spectral image in ``full`` mode.
        n = 24 if runner.mode == "quick" else 1024
        self._temperatures_np = np.linspace(2000, 10000, n)
        self._xy_d_series_np = self._rng.random((n, 2)) * 0.1 + (0.31, 0.33)
        self._xy_d_illuminant_np = np.array([0.31, 0.33])

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        return 6

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        temperatures = BenchmarkRunner.convert_to_backend(
            self._temperatures_np, backend
        )
        xy_d_series = BenchmarkRunner.convert_to_backend(self._xy_d_series_np, backend)

        yield (
            "sd_blackbody",
            {"function": "sd_blackbody"},
            lambda: sd_blackbody(6500, self._shape),
        )
        yield (
            "sd_rayleigh_jeans",
            {"function": "sd_rayleigh_jeans"},
            lambda: sd_rayleigh_jeans(6500, self._shape),
        )
        yield (
            "sd_CIE_illuminant_D_series",
            {"function": "sd_CIE_illuminant_D_series"},
            lambda x=self._xy_d_illuminant_np: sd_CIE_illuminant_D_series(x),
        )
        yield (
            "msds_blackbody",
            {"function": "msds_blackbody"},
            lambda t=temperatures: msds_blackbody(t, self._shape),
        )
        yield (
            "msds_rayleigh_jeans",
            {"function": "msds_rayleigh_jeans"},
            lambda t=temperatures: msds_rayleigh_jeans(t, self._shape),
        )
        yield (
            "msds_CIE_illuminant_D_series",
            {"function": "msds_CIE_illuminant_D_series"},
            lambda x=xy_d_series: msds_CIE_illuminant_D_series(x, shape=self._shape),
        )


class BenchmarkSuite_Photometry(BenchmarkSuite):
    """
    Benchmark photometric measures over a :class:`SpectralDistribution`.

    ``luminous_flux``, ``luminous_efficacy`` and ``luminous_efficiency``
    consume a :class:`SpectralDistribution` instance (numpy-backed) and
    are therefore numpy-bound regardless of the requested backend.
    """

    NAME = "photometry"

    def __init__(self, runner: BenchmarkRunner) -> None:
        super().__init__(runner)
        self._sd_template = SDS_LIGHT_SOURCES["Cool White FL"]

    @property
    def case_count(self) -> int:
        """See :class:`BenchmarkSuite`."""

        return 3

    def cases(
        self, backend: str
    ) -> Iterable[tuple[str, dict[str, str], Callable[[], object] | None]]:
        """See :class:`BenchmarkSuite`."""

        values = BenchmarkRunner.convert_to_backend(self._sd_template.values, backend)
        with array_api_enable(backend != "numpy"):
            sd_test = SpectralDistribution(
                values, self._sd_template.wavelengths, name=self._sd_template.name
            )

        yield (
            "luminous_flux",
            {"function": "luminous_flux"},
            lambda: luminous_flux(sd_test),
        )
        yield (
            "luminous_efficacy",
            {"function": "luminous_efficacy"},
            lambda: luminous_efficacy(sd_test),
        )
        yield (
            "luminous_efficiency",
            {"function": "luminous_efficiency"},
            lambda: luminous_efficiency(sd_test),
        )


BENCHMARK_SUITES: dict[str, type[BenchmarkSuite]] = {
    suite.NAME: suite
    for suite in (
        BenchmarkSuite_ConversionGraph,
        BenchmarkSuite_ConversionGraphIterative,
        BenchmarkSuite_Difference,
        BenchmarkSuite_IntegrationArray,
        BenchmarkSuite_IntegrationObject,
        BenchmarkSuite_TransferFunction,
        BenchmarkSuite_Adaptation,
        BenchmarkSuite_Characterisation,
        BenchmarkSuite_RecoveryArray,
        BenchmarkSuite_RecoveryObject,
        BenchmarkSuite_QualityArray,
        BenchmarkSuite_QualityObject,
        BenchmarkSuite_Volume,
        BenchmarkSuite_VolumeIterative,
        BenchmarkSuite_Phenomena,
        BenchmarkSuite_TemperatureArray,
        BenchmarkSuite_TemperatureIterative,
        BenchmarkSuite_Blindness,
        BenchmarkSuite_Contrast,
        BenchmarkSuite_GeneratorsArray,
        BenchmarkSuite_GeneratorsObject,
        BenchmarkSuite_Photometry,
    )
}
"""
Registry mapping suite names to their :class:`BenchmarkSuite` subclasses.

Listed explicitly: the dispatch order doubles as the suite execution order
when no ``--suites`` filter is provided.
"""


def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Cross-backend benchmarks for Colour.",
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="'quick' runs a correctness smoke test; 'full' runs HD timing.",
    )
    parser.add_argument(
        "--suites",
        default=",".join(BENCHMARK_SUITES),
        help="Comma-separated suite names. Available: " + ", ".join(BENCHMARK_SUITES),
    )
    parser.add_argument(
        "--backends",
        default=",".join(DEFAULT_BENCHMARK_CONFIGURATION.backends),
        help="Comma-separated backend list (unavailable backends are skipped).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Untimed iterations before timed runs.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Timed iterations; the best (minimum) is reported.",
    )
    parser.add_argument(
        "--edges",
        default=None,
        help='Conversion-graph edges filter, e.g. "CIE XYZ->CIE Lab,Src->Tgt".',
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV report path.",
    )
    parser.add_argument(
        "--log",
        default="benchmark.log",
        help="Log file path (default: CWD).",
    )
    return parser


def main() -> None:
    """Parse arguments, run the selected benchmarks, and emit reports."""

    args = build_argument_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.log, mode="w"),
            logging.StreamHandler(sys.stderr),
        ],
    )

    runner = BenchmarkRunner(
        mode=args.mode,
        backends=args.backends.split(","),
        warmup=args.warmup,
        runs=args.runs,
        edges_filter=args.edges,
    )

    suites = BenchmarkRunner.parse_suites(args.suites, runner)

    report = runner.run(suites)
    report.log()

    if args.json:
        write_report_json(report, args.json)

    if args.csv:
        write_report_csv(report, args.csv)


if __name__ == "__main__":
    main()
