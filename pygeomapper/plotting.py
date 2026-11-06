# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
Shared figure helpers for the pyGeoMapper notebooks and README figures (Matplotlib + Pillow only).

Includes:
- `save_figure`: single export path for every notebook figure — plain tick labels, tight white border,
  capped pixel width, Pillow-optimized PNG (optional 256-color palette) or JPEG; closes the figure so
  notebooks keep plots in `image/` only.
- `use_plain_axes`: plain-number map ticks (no `1e6` offset text) with optional km labels for meter grids.
- `drape_fringes`: wrapped InSAR phase fringes (from `surface_change.compute_insar_wrapped_phase` or a
  LiCSAR product) drawn over an RGB image or grayscale hillshade backdrop.
- `DOD_CMAP` / `dod_norm` / `DOD_LABEL`: the one DEM-of-Difference color rule used everywhere —
  red = erosion / loss (Δz < 0), blue = deposit / gain (Δz > 0).

Figures should be built with `layout="constrained"` (or `constrained_layout=True`) so titles, colorbars and
legends do not overlap; `save_figure` then only handles tick formatting, sizing and compression.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import numpy as np

DOD_CMAP = "RdBu"
DOD_LABEL = "Δz (m): red = erosion, blue = deposit"
FRINGE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "insar_fringe", ["#ff0000", "#ffff00", "#00ff00", "#00ffff", "#0000ff", "#ff00ff", "#ff0000"], N=256
)


def dod_norm(vmax: float, vmin: float | None = None) -> mcolors.Normalize:
    """Color norm for DoD panels, centered on 0 (symmetric `[-vmax, vmax]` unless `vmin` is given)."""
    lo = -abs(float(vmax)) if vmin is None else float(vmin)
    return mcolors.TwoSlopeNorm(vcenter=0.0, vmin=lo, vmax=abs(float(vmax)))


def colorbar_mappable(
    cmap: Any,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: mcolors.Normalize | None = None,
) -> Any:
    """Opaque `ScalarMappable` to feed `fig.colorbar` for pre-composited drapes.

    Exists so colorbars do not inherit overlay alpha: a hidden
    `imshow(..., alpha=0.0)` makes Matplotlib draw a blank colorbar.
    """
    import matplotlib.cm as mcm

    if norm is None:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm


def use_plain_axes(
    ax: Any,
    unit: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> Any:
    """
    Turn off Matplotlib offset / scientific tick text (the `1e6` corner label on UTM axes).

    `unit="km"` relabels meter-coordinate ticks in km (data and extent stay in meters); `unit="m"` keeps plain
    meters. When `xlabel` / `ylabel` are given they are written as `"<label> (<unit>)"`.
    """
    for axis in (ax.xaxis, ax.yaxis):
        if unit == "km":
            axis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v / 1000.0:g}"))
        else:
            fmt = axis.get_major_formatter()
            if isinstance(fmt, ScalarFormatter):
                fmt.set_useOffset(False)
                fmt.set_scientific(False)
    suffix = f" ({unit})" if unit else ""
    if xlabel:
        ax.set_xlabel(f"{xlabel}{suffix}")
    if ylabel:
        ax.set_ylabel(f"{ylabel}{suffix}")
    return ax


def _plain_offsets(fig: Any) -> None:
    for ax in fig.axes:
        if getattr(ax, "name", "") == "3d":
            continue
        for axis in (ax.xaxis, ax.yaxis):
            fmt = axis.get_major_formatter()
            if isinstance(fmt, ScalarFormatter):
                fmt.set_useOffset(False)
                fmt.set_scientific(False)


def _export_dpi(fig: Any, dpi: float, max_width_px: int | None, pad_inches: float) -> float:
    if not max_width_px:
        return float(dpi)
    try:
        width_in = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2.0 * pad_inches
    except Exception:  # pragma: no cover - non-Agg canvases
        width_in = fig.get_figwidth()
    return float(min(dpi, max_width_px / max(width_in, 1e-6)))


def _optimize_png(path: Path, palette: bool) -> None:
    from PIL import Image

    with Image.open(path) as im:
        im.load()
        rgba = im.convert("RGBA")
    rgb = Image.new("RGB", rgba.size, "white")
    rgb.paste(rgba, mask=rgba.split()[-1])
    if palette:
        quantize = getattr(Image, "Quantize", Image).MEDIANCUT
        dither = getattr(Image, "Dither", Image).NONE
        rgb = rgb.quantize(colors=256, method=quantize, dither=dither)
    rgb.save(path, optimize=True)


def save_figure(
    fig: Any,
    path: str | Path,
    dpi: float = 150.0,
    max_width_px: int | None = 2000,
    palette: bool = False,
    quality: int = 85,
    pad_inches: float = 0.12,
    close: bool = True,
) -> Path:
    """
    Save `fig` to `path` (`.png` or `.jpg`) on a white background with a tight border.

    - Disables tick offset / scientific text on every 2D axes so map panels never show a stray `1e6`.
    - Lowers the effective dpi so the saved image is at most `max_width_px` pixels wide.
    - PNG: re-encoded by Pillow as opaque RGB (`optimize=True`); `palette=True` reduces to 256 colors, which
      keeps noisy SAR / imagery boards around 1 MB. JPEG: saved at `quality`.
    - `close=True` closes the figure so executed notebooks keep plots on disk only.

    Returns the output `Path`.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _plain_offsets(fig)
    eff_dpi = _export_dpi(fig, dpi, max_width_px, pad_inches)
    kwargs: dict[str, Any] = dict(dpi=eff_dpi, bbox_inches="tight", pad_inches=pad_inches, facecolor="white")
    is_jpeg = out.suffix.lower() in (".jpg", ".jpeg")
    if is_jpeg:
        kwargs["pil_kwargs"] = {"quality": int(quality), "optimize": True, "progressive": True}
    fig.savefig(out, **kwargs)
    if out.suffix.lower() == ".png":
        _optimize_png(out, palette)
    if close:
        plt.close(fig)
    return out


def drape_fringes(
    ax: Any,
    wrapped_phase: np.ndarray,
    backdrop: np.ndarray,
    extent: tuple[float, float, float, float] | list[float],
    alpha: float = 0.60,
    backdrop_extent: tuple[float, float, float, float] | list[float] | None = None,
    cm_per_cycle: float | None = None,
    cmap: Any = None,
    aspect: str = "auto",
) -> Any:
    """
    Draw wrapped interferometric phase over an imagery or hillshade `backdrop` and return the fringe image
    (pass it to `fig.colorbar`).

    - `wrapped_phase`: radians in `[-pi, pi)` (e.g. `surface_change.compute_insar_wrapped_phase(los_m)`);
      NaN pixels stay transparent so the backdrop shows through.
    - `backdrop`: `(H, W, 3)` RGB / `(H, W, 4)` RGBA in `[0, 1]` (pre-event optical image; alpha 0 lets a layer
      drawn earlier, e.g. a hillshade, show through gaps) or `(H, W)` grayscale hillshade in `[0, 1]`.
    - `cm_per_cycle`: when given (Sentinel-1: 2.77 cm), the color axis shows LOS change in cm per fringe
      instead of radians.
    """
    bext = extent if backdrop_extent is None else backdrop_extent
    if np.ndim(backdrop) == 2:
        ax.imshow(backdrop, cmap="gray", vmin=0.0, vmax=1.0, extent=bext, origin="upper", aspect=aspect)
    else:
        ax.imshow(np.clip(backdrop, 0.0, 1.0), extent=bext, origin="upper", aspect=aspect)
    phase = np.asarray(wrapped_phase, dtype=np.float64)
    if cm_per_cycle is not None:
        values = np.mod(phase, 2.0 * np.pi) / (2.0 * np.pi) * float(cm_per_cycle)
        vmin, vmax = 0.0, float(cm_per_cycle)
    else:
        values, vmin, vmax = phase, -np.pi, np.pi
    return ax.imshow(
        np.ma.masked_invalid(values),
        cmap=FRINGE_CMAP if cmap is None else cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        extent=extent,
        origin="upper",
        aspect=aspect,
        interpolation="nearest",
    )
