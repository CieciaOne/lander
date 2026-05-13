"""Matplotlib plots for continual-learning evaluation results.

All plots share a thesis-grade style applied at module import: serif body
text (matches a LaTeX thesis at typesetting time), minimal spines, subtle
grid, a curated 8-colour cycle, and 200 DPI saved figures. Per-plot polish
on top: refined colormaps, annotated values, formatted sidebars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from rover_cl.eval.metrics import (
    EpisodeTrajectory,
    compute_avg_retention,
    compute_forgetting,
    compute_retention_matrix,
)

# 200 DPI is the rough sweet spot for thesis: crisp on screen, sharp at A4
# printing, file sizes still reasonable (~150–400 KB / plot).
_DPI = 200

# Curated 8-colour cycle. Inspired by 19th-century cartographic palettes —
# distinct in colour AND luminance (works in greyscale print), colourblind-safe
# pairwise except for a slight blue-green overlap. Avoid neon / pastel — both
# read as "casual" in thesis context.
_THESIS_COLORS = [
    "#2c5b8a",  # slate blue   (primary)
    "#c87533",  # burnt sienna (secondary)
    "#6b8e6a",  # sage green
    "#893945",  # wine
    "#c19a3b",  # ochre / mustard
    "#3a3a3a",  # charcoal
    "#7d6a93",  # dusty lavender
    "#4f9aa0",  # teal
]

# Per-episode colour palette for randomized-terrain reports. Each episode
# gets its own colour, and the path / start / goal / waypoints / obstacles
# of that episode all share it — so the reader can trace one rollout end
# to end even when 10 of them overlap. Picks distinct from each other in
# both hue and luminance; works on a white background.
_EPISODE_COLORS = [
    "#2c5b8a",  # slate
    "#c87533",  # sienna
    "#6b8e6a",  # sage
    "#893945",  # wine
    "#c19a3b",  # ochre
    "#7d6a93",  # lavender
    "#4f9aa0",  # teal
    "#a35d4b",  # terracotta
    "#3f6f4a",  # forest
    "#7d5a8b",  # plum
    "#b4624f",  # rust
    "#5f8aa0",  # steel
]


def _episode_speed_cmap(hex_color: str) -> "mpl.colors.LinearSegmentedColormap":
    """Build a per-episode colormap that fades from light (slow) to dark (fast).

    The episode is identified by hue (the base colour), and speed is encoded
    in lightness: slow segments are pale tints, fast segments are saturated.
    Stays unambiguous across 12 episode hues *and* preserves speed gradient
    information that the previous magma-uniform-across-episodes ramp lost.
    """
    import colorsys
    r, g, b, _ = mpl.colors.to_rgba(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    # Light end: lift lightness, drop saturation so it reads as "faded"
    light = colorsys.hls_to_rgb(h, min(0.92, l + 0.42), max(0.18, s - 0.25))
    # Dark end: saturate + darken slightly
    dark = colorsys.hls_to_rgb(h, max(0.18, l - 0.10), min(1.0, s + 0.05))
    return mpl.colors.LinearSegmentedColormap.from_list(
        f"ep_{hex_color}", [light, hex_color, dark], N=128,
    )

# Soft text/face colours used throughout — kept here so plot tweaks stay
# visually consistent across the module.
_FG = "#1d1d1d"
_FG_MUTED = "#5a5a5a"
_GRID = "#d9d9d9"
_PANEL_BG = "#f7f5f0"  # warm off-white for sidebars / stat boxes


def _apply_thesis_style() -> None:
    """Idempotent matplotlib rc tweaks. Applied at module import."""
    mpl.rcParams.update({
        # Typography: serif for body so the figure typesetting matches a
        # LaTeX thesis. macOS ships Charter; DejaVu Serif is the universal
        # matplotlib fallback. Sans-serif is reserved for fine annotations.
        "font.family": "serif",
        "font.serif": ["Charter", "Source Serif Pro", "Palatino",
                       "Times New Roman", "DejaVu Serif"],
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.0,
        "axes.titlesize": 11.0,
        "axes.titleweight": "regular",
        "axes.titlepad": 10.0,
        "axes.labelsize": 9.5,
        "axes.labelcolor": _FG,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.color": _FG_MUTED,
        "ytick.color": _FG_MUTED,
        "legend.fontsize": 8.5,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": _GRID,
        "legend.fancybox": False,
        # Spines: drop the top + right rules — feels modern, stays serious.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": _FG_MUTED,
        "axes.linewidth": 0.8,
        # Subtle grid only, drawn under everything.
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": _GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "grid.alpha": 0.65,
        # Line + marker defaults that scale well across plot sizes.
        "lines.linewidth": 1.8,
        "lines.markersize": 5.5,
        "patch.linewidth": 0.6,
        # Saving — keep this matched to _DPI.
        "savefig.dpi": _DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.10,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        # Color cycle.
        "axes.prop_cycle": mpl.cycler(color=_THESIS_COLORS),
    })


_apply_thesis_style()


def _save(fig: Figure, out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")


def plot_retention_matrix(
    retention: np.ndarray,
    task_ids: list[str],
    title: str,
    out: Path,
) -> Figure:
    """Heatmap of success_rate over training phases x evaluation tasks.

    Rows correspond to phases (after training task k); columns are evaluation
    tasks. NaN cells (not yet trained on) are drawn in gray.
    """
    retention = np.asarray(retention, dtype=float)
    n_rows, n_cols = retention.shape

    fig, ax = plt.subplots(figsize=(max(4.8, 0.85 * n_cols + 2.8),
                                    max(4.2, 0.7 * n_rows + 2.2)))

    # `crest` from seaborn-style ramps would be nicer but isn't always available;
    # `mako` ships with matplotlib's perceptually-uniform set. NaN cells (tasks
    # not yet evaluated) render in soft warm grey so they read as "missing"
    # rather than zero.
    cmap_name = "mako" if "mako" in mpl.colormaps else "viridis"
    cmap = mpl.colormaps[cmap_name].with_extremes(bad="#e8e2d6")
    masked = np.ma.masked_invalid(retention)
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    # Spines disabled by default style; for heatmaps we want a thin frame.
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(_GRID)

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(task_ids, fontsize=8.5)
    ax.set_yticklabels([f"after {task_ids[k]}" for k in range(n_rows)],
                       fontsize=8.5)
    ax.set_xlabel("Evaluation task", color=_FG_MUTED)
    ax.set_ylabel("Training phase", color=_FG_MUTED)
    ax.set_title(title, color=_FG, loc="left")
    ax.grid(False)

    # White inter-cell separators give the heatmap a tiled feel without the
    # noisy "borders on every cell" look.
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    for i in range(n_rows):
        for j in range(n_cols):
            value = retention[i, j]
            if np.isnan(value):
                ax.text(j, i, "—", ha="center", va="center",
                        color=_FG_MUTED, fontsize=9, family="sans-serif")
            else:
                # Pick text colour so it stays legible across the colormap.
                color = "white" if value < 0.55 else "#0d1b2a"
                ax.text(
                    j, i, f"{value:.2f}",
                    ha="center", va="center", color=color,
                    fontsize=9.5, family="sans-serif", weight="medium",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.044, pad=0.04)
    cbar.set_label("success rate", color=_FG_MUTED)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, color=_FG_MUTED)

    fig.tight_layout()
    _save(fig, out)
    return fig


def plot_retention_curves(
    results: dict[str, Any],
    task_ids: list[str],
    out: Path,
) -> Figure:
    """One line per task: success_rate as the curriculum advances."""
    retention = compute_retention_matrix(results)
    n_rows, n_cols = retention.shape
    phases = np.arange(n_rows)

    fig, ax = plt.subplots(figsize=(max(5.4, 0.9 * n_rows + 3.2), 4.2))

    for j in range(n_cols):
        col = retention[:, j]
        valid = ~np.isnan(col)
        if not valid.any():
            continue
        color = _THESIS_COLORS[j % len(_THESIS_COLORS)]
        # Light "shadow" stroke under the main line gives the figure a bit of
        # presence without being noisy — common trick in thesis-style plots.
        ax.plot(
            phases[valid], col[valid],
            color=color, linewidth=4.0, alpha=0.12, solid_capstyle="round",
        )
        ax.plot(
            phases[valid], col[valid],
            color=color, linewidth=2.0,
            marker="o", markersize=6.5, markeredgewidth=0.0,
            label=task_ids[j], solid_capstyle="round",
        )
        # Annotate the final value so readers don't have to eyeball the y-axis.
        last_phase = phases[valid].max()
        last_val = col[valid][-1]
        ax.annotate(
            f"{last_val:.2f}",
            xy=(last_phase, last_val),
            xytext=(6, 0), textcoords="offset points",
            color=color, fontsize=8.5, family="sans-serif", va="center",
        )

    ax.set_xticks(phases)
    ax.set_xticklabels([f"after {task_ids[k]}" for k in range(n_rows)],
                       rotation=15, ha="right")
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlim(-0.3, n_rows - 0.55)  # leave room on the right for annotations
    ax.set_xlabel("Training phase", color=_FG_MUTED)
    ax.set_ylabel("success rate", color=_FG_MUTED)
    method = results.get("cl_method", "")
    mission = results.get("mission_name", "")
    title_bits = [b for b in [mission, method] if b]
    title = "Retention per task" + (f"  ·  {' | '.join(title_bits)}" if title_bits else "")
    ax.set_title(title, color=_FG, loc="left")
    ax.legend(title="task", loc="lower left", title_fontsize=8.5)

    fig.tight_layout()
    _save(fig, out)
    return fig


def plot_method_comparison_with_variance(
    results_by_method: dict[str, list[dict[str, Any]]],
    task_ids: list[str],
    out: Path,
    metric: str = "avg_retention",
) -> Figure:
    """Bar chart: per-method mean of ``metric`` with +/- std error bars across seeds.

    Parameters
    ----------
    results_by_method:
        Maps a method name to a list of per-seed results dicts (each one the
        same shape as a Runner ``results.json``).
    task_ids:
        Used only for the title's task count.
    metric:
        ``"avg_retention"`` or ``"forgetting"``.
    """
    if metric not in {"avg_retention", "forgetting"}:
        raise ValueError(
            f"unknown metric {metric!r}; expected 'avg_retention' or 'forgetting'"
        )

    methods = list(results_by_method.keys())
    means: list[float] = []
    stds: list[float] = []
    n_seeds_per_method: list[int] = []

    for method in methods:
        seed_results = results_by_method[method]
        n_seeds_per_method.append(len(seed_results))
        per_seed_values: list[float] = []
        for res in seed_results:
            retention = compute_retention_matrix(res)
            if metric == "avg_retention":
                per_seed_values.append(compute_avg_retention(retention))
            else:
                forgetting = compute_forgetting(retention)
                per_seed_values.append(
                    float(np.nanmean(forgetting)) if forgetting.size else float("nan")
                )
        arr = np.asarray(per_seed_values, dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            means.append(float("nan"))
            stds.append(float("nan"))
        else:
            means.append(float(np.nanmean(arr)))
            stds.append(float(np.nanstd(arr, ddof=0)))

    # Sanitize NaNs for plotting (matplotlib can't draw NaN bars cleanly).
    plot_means = [0.0 if np.isnan(v) else v for v in means]
    plot_stds = [0.0 if np.isnan(v) else v for v in stds]

    fig, ax = plt.subplots(figsize=(max(4.8, 1.3 * len(methods) + 2.6), 4.4))
    colors = [_THESIS_COLORS[i % len(_THESIS_COLORS)] for i in range(len(methods))]
    x = np.arange(len(methods))
    bars = ax.bar(
        x,
        plot_means,
        yerr=plot_stds,
        color=colors,
        edgecolor="white",
        linewidth=1.0,
        width=0.62,
        capsize=4,
        error_kw={"elinewidth": 1.4, "ecolor": _FG, "capthick": 1.4},
    )
    # Subtle bottom-baseline accent — a thin horizontal rule at y=0 grounds
    # the bars without the boxy "full frame" look.
    ax.axhline(0.0, color=_FG_MUTED, linewidth=0.7, zorder=0)

    for xi, mean_val, std_val, n_seeds, bar in zip(x, means, stds, n_seeds_per_method, bars):
        if np.isnan(mean_val):
            label = "n/a"
        else:
            label = f"{mean_val:.2f}"
        y_top = (0.0 if np.isnan(mean_val) else mean_val) + (
            0.0 if np.isnan(std_val) else std_val
        )
        ax.text(
            xi, y_top + 0.025,
            label,
            ha="center", va="bottom",
            fontsize=10, family="sans-serif", weight="medium", color=_FG,
        )
        # Seed count goes inside the bar, more discreet than stacking on top.
        if not np.isnan(mean_val) and mean_val > 0.08:
            ax.text(
                xi, mean_val * 0.5,
                f"n={n_seeds}",
                ha="center", va="center",
                fontsize=7.5, family="sans-serif",
                color="white", alpha=0.85,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_xlabel("CL method", color=_FG_MUTED)
    ax.grid(True, axis="y", linewidth=0.5, color=_GRID, alpha=0.65)
    ax.grid(False, axis="x")

    # Common seed count for the title if all methods used the same N seeds;
    # otherwise show the range.
    if n_seeds_per_method:
        n_min = min(n_seeds_per_method)
        n_max = max(n_seeds_per_method)
        n_label = f"N={n_min}" if n_min == n_max else f"N={n_min}..{n_max}"
    else:
        n_label = "N=0"

    mission = ""
    for seed_list in results_by_method.values():
        if seed_list:
            mission = seed_list[0].get("mission_name", "") or ""
            break
    title_prefix = mission if mission else "comparison"

    if metric == "avg_retention":
        ax.set_ylabel("avg success_rate (final phase)")
        ax.set_ylim(0.0, 1.2)
        ax.set_title(
            f"{title_prefix} — avg retention ({n_label} seeds, "
            f"{len(task_ids)} tasks)"
        )
    else:
        ax.set_ylabel("mean forgetting")
        finite = [m + s for m, s in zip(plot_means, plot_stds) if not np.isnan(m)]
        upper = max(finite + [0.0]) * 1.3 + 0.05 if finite else 1.0
        ax.set_ylim(0.0, max(upper, 0.1))
        ax.set_title(
            f"{title_prefix} — mean forgetting ({n_label} seeds, "
            f"{len(task_ids)} tasks)"
        )
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, out)
    return fig


def plot_method_comparison(
    results_by_method: dict[str, dict[str, Any]],
    task_ids: list[str],
    out: Path,
    metric: str = "avg_retention",
) -> Figure:
    """Bar chart comparing CL methods on a chosen aggregate metric."""
    if metric not in {"avg_retention", "forgetting"}:
        raise ValueError(
            f"unknown metric {metric!r}; expected 'avg_retention' or 'forgetting'"
        )

    methods = list(results_by_method.keys())
    values: list[float] = []
    for method in methods:
        retention = compute_retention_matrix(results_by_method[method])
        if metric == "avg_retention":
            values.append(compute_avg_retention(retention))
        else:
            forgetting = compute_forgetting(retention)
            values.append(float(np.nanmean(forgetting)) if forgetting.size else float("nan"))

    fig, ax = plt.subplots(figsize=(max(4.5, 1.1 * len(methods) + 2.5), 4.0))
    cmap = mpl.colormaps["tab10"]
    colors = [cmap(i % 10) for i in range(len(methods))]
    x = np.arange(len(methods))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6, width=0.65)

    for bar, value in zip(bars, values):
        if np.isnan(value):
            label = "n/a"
        else:
            label = f"{value:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.01 if not np.isnan(bar.get_height()) else 0.01),
            label,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_xlabel("CL method")
    if metric == "avg_retention":
        ax.set_ylabel("avg success_rate (final phase)")
        ax.set_ylim(0.0, 1.1)
        ax.set_title(f"Average retention across {len(task_ids)} tasks")
    else:
        ax.set_ylabel("mean forgetting")
        finite = [v for v in values if not np.isnan(v)]
        upper = max(finite + [0.0]) * 1.25 + 0.05 if finite else 1.0
        ax.set_ylim(0.0, max(upper, 0.1))
        ax.set_title(f"Mean forgetting across {len(task_ids)} tasks")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, out)
    return fig


def plot_run_report(
    terrain: Any,
    trajectories: list[EpisodeTrajectory],
    out: Path,
    title: str,
    *,
    max_drawn_trajectories: int | None = 5,
) -> Figure:
    """Top-down report: terrain + obstacles + start/waypoints/goal + paths + hits.

    Each trajectory is drawn as a line colored by outcome (green = success,
    red = tipped, orange = timeout). Contact-step positions scatter as red X.
    A sidebar reports aggregate stats so the image is a standalone report.

    `max_drawn_trajectories` (default 5) ranks trajectories — successes first,
    then by `final_distance_to_goal` ascending — and renders only the top N
    paths / obstacles / waypoints / goals. Stats in the sidebar are still
    computed across ALL trajectories (you want the true success rate, not the
    top-N's). Pass `None` to draw everything.
    """
    # Rank trajectories by success then proximity-to-goal; keep the original
    # full list for stats so the sidebar percentages stay honest.
    all_trajectories = trajectories
    if max_drawn_trajectories is not None and len(trajectories) > max_drawn_trajectories:
        def _sort_key(t: EpisodeTrajectory) -> tuple[int, float]:
            # success=True comes first (0 < 1 when reversed via `not`);
            # then nearer-to-goal first.
            return (0 if t.success else 1, float(t.final_distance_to_goal))
        trajectories = sorted(trajectories, key=_sort_key)[:max_drawn_trajectories]
    fig, (ax, ax_stats) = plt.subplots(
        ncols=2,
        figsize=(12.0, 7.6),
        gridspec_kw={"width_ratios": [3.2, 1.0], "wspace": 0.18},
    )
    # A faint warm tint on the figure background sets the report apart from
    # generic matplotlib — common in published cartography / GIS figures.
    fig.patch.set_facecolor("#fbfaf6")

    # Thesis palette colours used throughout this plot, named for clarity.
    C_START = _THESIS_COLORS[0]      # slate blue
    C_GOAL = _THESIS_COLORS[2]       # sage green
    C_WAYPOINT = "#3a78a8"            # softer blue than start so they don't clash
    C_OBSTACLE = "#544840"
    C_OBSTACLE_EDGE = "#2c241d"
    C_CONTACT = _THESIS_COLORS[3]    # wine for contact crosses

    half = float(terrain.arena_half_extent)
    margin = 0.5
    ax.set_xlim(-half - margin, half + margin)
    ax.set_ylim(-half - margin, half + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color=_FG_MUTED)
    ax.set_ylabel("y (m)", color=_FG_MUTED)
    if len(trajectories) < len(all_trajectories):
        full_title = (f"{title}    "
                      f"·  showing {len(trajectories)} best of "
                      f"{len(all_trajectories)} episodes")
    else:
        full_title = title
    ax.set_title(full_title, color=_FG, loc="left", fontsize=10.5)
    # Suppress the global grid; replace with a softer minor grid below.
    ax.grid(False)
    ax.set_xticks(np.arange(-int(half), int(half) + 1, 5))
    ax.set_yticks(np.arange(-int(half), int(half) + 1, 5))
    ax.set_xticks(np.arange(-int(half), int(half) + 1, 1), minor=True)
    ax.set_yticks(np.arange(-int(half), int(half) + 1, 1), minor=True)
    ax.grid(which="major", color=_GRID, linewidth=0.45, alpha=0.85, zorder=1)
    ax.grid(which="minor", color=_GRID, linewidth=0.25, alpha=0.4, zorder=1)
    # Keep a thin frame on this plot — it's a map, not a chart.
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(_FG_MUTED)
        ax.spines[side].set_linewidth(0.7)

    # Heightmap background, if any. `gist_earth` reads as a real-terrain map
    # (low = water-ish, high = ridge-ish) and pairs cleanly with the warm
    # figure background. Alpha low so the trajectory still pops on top.
    hfield_cbar = None
    hm = getattr(terrain, "heightmap", None)
    if hm is not None:
        rx, ry, ez = terrain.heightmap_extent
        elev_m = hm * float(ez)
        im = ax.imshow(
            elev_m,
            extent=(-rx, rx, -ry, ry),
            origin="lower",
            cmap="gist_earth",
            alpha=0.42,
            zorder=0,
            vmin=0.0,
            vmax=float(ez) if float(ez) > 0 else 1.0,
        )
        hfield_cbar = fig.colorbar(im, ax=ax, fraction=0.033, pad=0.02)
        hfield_cbar.set_label("terrain height (m)", color=_FG_MUTED, fontsize=8)
        hfield_cbar.outline.set_visible(False)
        hfield_cbar.ax.tick_params(labelsize=7, color=_FG_MUTED)

    # Arena boundary (dashed) — only when no heightmap, otherwise the
    # heightmap edge already implies the bounds.
    if hm is None:
        ax.add_patch(mpl.patches.Rectangle(
            (-half, -half), 2 * half, 2 * half,
            fill=False, edgecolor=_FG_MUTED, linewidth=0.8, linestyle=(0, (5, 4)),
        ))

    # Detect "randomized mode": at least one trajectory carries its own
    # per-episode obstacle layout OR per-episode start / goal pose. When in
    # randomized mode, EVERY visual element belonging to episode `i` (path,
    # obstacles, start, goal, waypoints) is drawn in that episode's distinct
    # colour, so the reader can trace one rollout end-to-end through the
    # overlap. In non-randomized mode, the original single-layout style is
    # preserved and paths are speed-coloured.
    randomized_mode = any(
        (tr.obstacle_layout is not None and tr.obstacle_layout.shape[0] > 0)
        or tr.start_pos is not None
        for tr in trajectories
    )

    def _ep_color(i: int) -> str:
        return _EPISODE_COLORS[i % len(_EPISODE_COLORS)]

    if randomized_mode:
        # Draw each episode's obstacles in that episode's colour, low alpha.
        for i, tr in enumerate(trajectories):
            ec = _ep_color(i)
            if tr.obstacle_layout is None:
                continue
            for cx, cy, sx, sy in tr.obstacle_layout:
                ax.add_patch(mpl.patches.Rectangle(
                    (cx - sx, cy - sy), 2 * sx, 2 * sy,
                    facecolor=ec, edgecolor=ec,
                    linewidth=0.5, alpha=0.28, zorder=2,
                ))
    else:
        for ob in terrain.obstacles:
            cx, cy = ob.pos[0], ob.pos[1]
            sx, sy = ob.size[0], ob.size[1]
            # shadow
            ax.add_patch(mpl.patches.Rectangle(
                (cx - sx + 0.06, cy - sy - 0.06), 2 * sx, 2 * sy,
                facecolor="black", alpha=0.12, linewidth=0, zorder=1.4,
            ))
            # body
            ax.add_patch(mpl.patches.Rectangle(
                (cx - sx, cy - sy), 2 * sx, 2 * sy,
                facecolor=C_OBSTACLE, edgecolor=C_OBSTACLE_EDGE, linewidth=0.9,
                alpha=0.92, zorder=2,
            ))

    # Start / waypoint / goal markers.
    if randomized_mode:
        for i, tr in enumerate(trajectories):
            ec = _ep_color(i)
            # Start: ring (no fill) + small filled dot + numeric label.
            if tr.start_pos is not None:
                sx, sy = tr.start_pos
                ax.add_patch(mpl.patches.Circle(
                    (sx, sy), 0.55,
                    facecolor="none", edgecolor=ec, linewidth=1.4,
                    alpha=0.85, zorder=4,
                ))
                ax.plot(sx, sy, marker="o", markersize=6, markerfacecolor=ec,
                        markeredgecolor="white", markeredgewidth=1.0,
                        linestyle="none", zorder=5)
                # Episode-index pill so overlaps stay traceable.
                ax.annotate(
                    str(i),
                    (sx, sy), textcoords="offset points", xytext=(8, 8),
                    fontsize=8, family="sans-serif", color="white",
                    weight="bold", ha="center", va="center",
                    bbox={
                        "facecolor": ec, "edgecolor": "none",
                        "boxstyle": "circle,pad=0.18", "alpha": 0.92,
                    },
                    zorder=6,
                )
            # Goal: star + halo in the same colour.
            if tr.goal_pos is not None:
                gx, gy = tr.goal_pos
                ax.add_patch(mpl.patches.Circle(
                    (gx, gy), float(terrain.goal_radius),
                    facecolor=ec, edgecolor=ec, linewidth=0.6,
                    alpha=0.18, zorder=3,
                ))
                ax.plot(gx, gy, marker="*", markersize=13, markerfacecolor=ec,
                        markeredgecolor="white", markeredgewidth=1.0,
                        linestyle="none", zorder=5)
            # Waypoints: rings only (no halo) to keep clutter down.
            for (wx, wy) in tr.waypoints:
                ax.add_patch(mpl.patches.Circle(
                    (wx, wy), getattr(terrain, "waypoint_radius", 1.5),
                    facecolor="none", edgecolor=ec, linewidth=0.9,
                    alpha=0.45, zorder=3,
                ))
                ax.plot(wx, wy, marker="o", markersize=4,
                        markerfacecolor=ec, markeredgecolor="white",
                        markeredgewidth=0.7,
                        linestyle="none", zorder=5)
    else:
        sx, sy = terrain.start_pos
        ax.add_patch(mpl.patches.Circle(
            (sx, sy), 0.55, facecolor=C_START, alpha=0.18, edgecolor="none", zorder=4,
        ))
        ax.plot(sx, sy, marker="o", markersize=9, markerfacecolor=C_START,
                markeredgecolor="white", markeredgewidth=1.4,
                linestyle="none", zorder=5)
        for i, (wx, wy) in enumerate(getattr(terrain, "waypoints", ()) or ()):
            ax.add_patch(mpl.patches.Circle(
                (wx, wy), getattr(terrain, "waypoint_radius", 1.5),
                facecolor=C_WAYPOINT, edgecolor=C_WAYPOINT, linewidth=0.8,
                alpha=0.20, zorder=3,
            ))
            ax.plot(wx, wy, marker="o", markersize=7, markerfacecolor=C_WAYPOINT,
                    markeredgecolor="white", markeredgewidth=1.2,
                    linestyle="none", zorder=5)
            ax.annotate(
                f"wp{i}", (wx, wy),
                textcoords="offset points", xytext=(9, 8),
                fontsize=8.5, color=_FG, family="sans-serif",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.5},
            )
        gx, gy = terrain.goal_pos
        ax.add_patch(mpl.patches.Circle(
            (gx, gy), float(terrain.goal_radius),
            facecolor=C_GOAL, edgecolor=C_GOAL, linewidth=0.8,
            alpha=0.22, zorder=3,
        ))
        ax.plot(gx, gy, marker="*", markersize=16, markerfacecolor=C_GOAL,
                markeredgecolor="white", markeredgewidth=1.2,
                linestyle="none", zorder=5)

    # Pre-compute per-segment speed across all trajectories so the colormap is
    # normalized over the full range, not per-episode (which would make slow
    # and fast episodes look identical).
    from matplotlib.collections import LineCollection
    DT_PER_STEP = 0.025  # env step seconds (timestep * control_decimation)
    max_speed_observed = 0.0
    per_traj_segments_and_speeds: list[tuple[np.ndarray, np.ndarray]] = []
    for tr in trajectories:
        if tr.positions.shape[0] < 2:
            per_traj_segments_and_speeds.append((np.zeros((0, 2, 2), dtype=np.float32),
                                                 np.zeros(0, dtype=np.float32)))
            continue
        pts = tr.positions.reshape(-1, 1, 2)
        seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
        speeds = np.linalg.norm(np.diff(tr.positions, axis=0), axis=1) / DT_PER_STEP
        max_speed_observed = max(max_speed_observed, float(speeds.max(initial=0.0)))
        per_traj_segments_and_speeds.append((seg, speeds))
    # cap the colormap upper bound: gives consistent meaning across episodes
    # even when one outlier hits an unrealistic spike.
    vmax = max(0.6, min(max_speed_observed, 1.5))  # m/s
    norm_speed = mpl.colors.Normalize(vmin=0.0, vmax=vmax)

    outcome_color = {
        "success": _THESIS_COLORS[2],   # sage green
        "tipped": _THESIS_COLORS[3],    # wine
        "timeout": _THESIS_COLORS[1],   # burnt sienna
    }
    seen_outcomes: set[str] = set()
    all_contacts: list[tuple[float, float, str]] = []  # (x, y, episode_color)
    speed_artist = None
    SPEED_CMAP = "magma"
    for i, (tr, (seg, speeds)) in enumerate(zip(trajectories, per_traj_segments_and_speeds)):
        if tr.positions.shape[0] == 0:
            continue

        if randomized_mode:
            # Per-episode hue identifies the rollout; lightness encodes speed.
            # Light = slow, saturated = fast. Lets the reader trace one episode
            # end-to-end AND see where it accelerated / stalled.
            ec = _ep_color(i)
            # White underlay for legibility against the heightmap.
            ax.plot(tr.positions[:, 0], tr.positions[:, 1],
                    color="white", linewidth=3.6, alpha=0.6,
                    solid_capstyle="round", zorder=2.5)
            if seg.shape[0] > 0:
                ep_cmap = _episode_speed_cmap(ec)
                lc = LineCollection(seg, cmap=ep_cmap, norm=norm_speed,
                                    linewidths=2.1, alpha=0.95,
                                    capstyle="round", zorder=3)
                lc.set_array(speeds)
                ax.add_collection(lc)
            end_face = ec
            end_edge_color = outcome_color[tr.outcome]  # outcome shows in ring
        else:
            # Speed-coloured render for fixed terrains.
            c_out = outcome_color[tr.outcome]
            if seg.shape[0] > 0:
                ax.plot(tr.positions[:, 0], tr.positions[:, 1],
                        color="white", linewidth=3.6, alpha=0.55,
                        solid_capstyle="round", zorder=2.5)
                lc = LineCollection(seg, cmap=SPEED_CMAP, norm=norm_speed,
                                    linewidths=1.9, alpha=0.95,
                                    capstyle="round", zorder=3)
                lc.set_array(speeds)
                ax.add_collection(lc)
                speed_artist = lc
            end_face = c_out
            end_edge_color = "white"

        # Endpoint dot — outcome lives in the ring colour when in randomized
        # mode (so the ring tells you success/timeout/tipped while the fill
        # matches the path).
        ax.plot(tr.positions[-1, 0], tr.positions[-1, 1],
                marker="o", markersize=6.5, markerfacecolor=end_face,
                markeredgecolor=end_edge_color, markeredgewidth=1.4,
                linestyle="none", zorder=4)
        seen_outcomes.add(tr.outcome)
        if tr.contact_positions.shape[0] > 0:
            ec_for_contacts = _ep_color(i) if randomized_mode else C_CONTACT
            for cx, cy in tr.contact_positions.tolist():
                all_contacts.append((float(cx), float(cy), ec_for_contacts))

    if speed_artist is not None and not randomized_mode:
        cbar = fig.colorbar(
            speed_artist, ax=ax,
            fraction=0.033, pad=0.02 if hfield_cbar is None else 0.10,
        )
        cbar.set_label("rover speed (m/s)", color=_FG_MUTED, fontsize=8)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=7, color=_FG_MUTED)

    if all_contacts:
        # Contacts are X-marks coloured per-episode in randomized mode (so a
        # contact belongs to a specific path), else single C_CONTACT colour.
        xs = [c[0] for c in all_contacts]
        ys = [c[1] for c in all_contacts]
        colors = [c[2] for c in all_contacts]
        ax.scatter(xs, ys, marker="x", s=26, c=colors, linewidths=1.4, zorder=6)

    legend_handles: list = []
    if randomized_mode:
        legend_handles += [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_FG_MUTED,
                       markeredgecolor="white", markeredgewidth=1.0,
                       markersize=7, label="start (number = episode)"),
            plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=_FG_MUTED,
                       markeredgecolor="white", markeredgewidth=1.0,
                       markersize=12, label="goal"),
        ]
        for name in sorted(seen_outcomes):
            legend_handles.append(plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor="white", markeredgecolor=outcome_color[name],
                markeredgewidth=1.6, markersize=8,
                label=f"end ring: {name}",
            ))
    else:
        legend_handles += [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_START,
                       markeredgecolor="white", markeredgewidth=1.2,
                       markersize=8, label="start"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_WAYPOINT,
                       markeredgecolor="white", markeredgewidth=1.2,
                       markersize=7, label="waypoint"),
            plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=C_GOAL,
                       markeredgecolor="white", markeredgewidth=1.2,
                       markersize=12, label="goal"),
        ]
        for name in sorted(seen_outcomes):
            legend_handles.append(plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=outcome_color[name], markeredgecolor="white",
                markeredgewidth=1.0, markersize=7,
                label=f"end: {name}",
            ))
    if all_contacts:
        legend_handles.append(plt.Line2D(
            [0], [0], marker="x", color=_FG_MUTED,
            linestyle="none", markersize=8, markeredgewidth=1.4,
            label="contact",
        ))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              framealpha=0.94, facecolor="white", edgecolor=_GRID)

    ax_stats.axis("off")
    # Stats reflect ALL trajectories (true success rate, etc.) — the
    # drawn-on-map sample is just a top-N view for readability.
    n = len(all_trajectories)
    if n > 0:
        n_success = sum(1 for t in all_trajectories if t.success)
        n_tipped = sum(1 for t in all_trajectories if t.tipped)
        n_timeout = sum(1 for t in all_trajectories if not t.success and not t.tipped)
        n_hit = sum(1 for t in all_trajectories if t.contact_positions.shape[0] > 0)
        steps_success = [t.steps for t in all_trajectories if t.success]
        steps_all = [t.steps for t in all_trajectories]
        mean_steps_success = float(np.mean(steps_success)) if steps_success else float("nan")
        mean_steps_all = float(np.mean(steps_all)) if steps_all else float("nan")
        mean_d = float(np.mean([t.final_distance_to_goal for t in all_trajectories]))
        mean_r = float(np.mean([t.cumulative_reward for t in all_trajectories]))
        mean_contact = float(np.mean([t.contact_positions.shape[0] for t in all_trajectories]))
        # Speed analysis: per-step speed = ‖Δpos‖ / dt where dt = 0.025 s.
        # Aggregate across all per-step speeds in every episode.
        DT = 0.025
        all_speeds: list[float] = []
        stalled_step_frac_per_ep: list[float] = []
        for t in all_trajectories:
            if t.positions.shape[0] < 2:
                continue
            sp = np.linalg.norm(np.diff(t.positions, axis=0), axis=1) / DT
            all_speeds.extend(sp.tolist())
            # "Stalled" = below 0.05 m/s (rover essentially not moving).
            stalled_step_frac_per_ep.append(float((sp < 0.05).mean()))
        if all_speeds:
            mean_speed = float(np.mean(all_speeds))
            max_speed = float(np.max(all_speeds))
            mean_stalled_frac = (
                float(np.mean(stalled_step_frac_per_ep))
                if stalled_step_frac_per_ep else 0.0
            )
        else:
            mean_speed = max_speed = mean_stalled_frac = float("nan")
    else:
        n_success = n_tipped = n_timeout = n_hit = 0
        mean_steps_success = float("nan")
        mean_steps_all = float("nan")
        mean_d = float("nan")
        mean_r = float("nan")
        mean_contact = float("nan")
        mean_speed = float("nan")
        max_speed = float("nan")
        mean_stalled_frac = float("nan")

    success_rate = (n_success / n) if n else 0.0

    # Panel background — gives the sidebar visual weight and groups the stats
    # together. Uses figure coords so it sits behind the text we place below.
    ax_stats.add_patch(mpl.patches.FancyBboxPatch(
        (0.02, 0.04), 0.96, 0.94,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        transform=ax_stats.transAxes,
        facecolor=_PANEL_BG, edgecolor=_GRID, linewidth=0.8, zorder=0,
    ))

    def _stat_color(metric: str, val: float) -> str:
        """Light editorial colouring on key stats — green for good, red for bad."""
        if metric == "success" and n:
            if val >= 0.7: return _THESIS_COLORS[2]   # sage
            if val >= 0.3: return _THESIS_COLORS[4]   # ochre
            return _THESIS_COLORS[3]                  # wine
        if metric == "hits":
            if val == 0: return _THESIS_COLORS[2]
            if val <= n * 0.3: return _THESIS_COLORS[4]
            return _THESIS_COLORS[3]
        return _FG

    # Big "headline" success rate at the top of the panel.
    ax_stats.text(
        0.5, 0.93,
        f"{success_rate * 100:.0f}%",
        ha="center", va="top",
        family="serif", fontsize=32, weight="bold",
        color=_stat_color("success", success_rate),
        transform=ax_stats.transAxes,
    )
    ax_stats.text(
        0.5, 0.78,
        f"success rate · n={n}",
        ha="center", va="top",
        family="sans-serif", fontsize=9, color=_FG_MUTED,
        transform=ax_stats.transAxes,
    )
    # Thin divider.
    ax_stats.plot([0.10, 0.90], [0.74, 0.74],
                  color=_GRID, linewidth=0.8, transform=ax_stats.transAxes)

    # Outcome breakdown — three little stat tiles in a row.
    tile_y = 0.66
    tile_specs = [
        ("success", n_success, _THESIS_COLORS[2]),
        ("tipped",  n_tipped,  _THESIS_COLORS[3]),
        ("timeout", n_timeout, _THESIS_COLORS[1]),
    ]
    for i, (label, val, color) in enumerate(tile_specs):
        cx = 0.18 + i * 0.32
        ax_stats.text(
            cx, tile_y, str(val),
            ha="center", va="center",
            family="serif", fontsize=18, weight="bold", color=color,
            transform=ax_stats.transAxes,
        )
        ax_stats.text(
            cx, tile_y - 0.08, label,
            ha="center", va="center",
            family="sans-serif", fontsize=8, color=_FG_MUTED,
            transform=ax_stats.transAxes,
        )

    # Bottom: per-row key/value list (typeset, not monospace dump).
    def _fmt(v: float, fmt: str, na: str = "n/a") -> str:
        return na if (isinstance(v, float) and np.isnan(v)) else format(v, fmt)

    rows = [
        ("steps (success)", _fmt(mean_steps_success, ".0f")),
        ("steps (all)",     _fmt(mean_steps_all, ".0f")),
        ("final d to goal", _fmt(mean_d, ".2f") + " m"),
        ("mean return",     _fmt(mean_r, "+.2f")),
        ("mean speed",      _fmt(mean_speed, ".2f") + " m/s"),
        ("peak speed",      _fmt(max_speed, ".2f") + " m/s"),
        ("stalled steps %", _fmt(mean_stalled_frac * 100, ".0f") + " %"
                            if not np.isnan(mean_stalled_frac) else "n/a"),
        ("contact steps",   _fmt(mean_contact, ".1f")),
        ("any-hit episodes", f"{n_hit}/{n}"),
    ]
    row_y0 = 0.46
    row_dy = 0.048
    ax_stats.text(
        0.5, row_y0 + 0.05,
        "averages across episodes",
        ha="center", va="bottom",
        family="sans-serif", fontsize=8, color=_FG_MUTED, style="italic",
        transform=ax_stats.transAxes,
    )
    for i, (k, v) in enumerate(rows):
        y = row_y0 - i * row_dy
        ax_stats.text(
            0.12, y, k,
            ha="left", va="center",
            family="sans-serif", fontsize=8.5, color=_FG_MUTED,
            transform=ax_stats.transAxes,
        )
        # Slight emphasis on "any-hit" since it's the recovery-failure signal.
        is_hit = (k == "any-hit episodes")
        ax_stats.text(
            0.88, y, v,
            ha="right", va="center",
            family="sans-serif", fontsize=9.5,
            weight="medium" if is_hit else "regular",
            color=(_stat_color("hits", n_hit) if is_hit else _FG),
            transform=ax_stats.transAxes,
        )

    # tight_layout doesn't play well with the FancyBboxPatch in axes coords —
    # use a manual subplots_adjust instead to keep the layout deterministic.
    fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.08, wspace=0.18)
    _save(fig, out)
    return fig
