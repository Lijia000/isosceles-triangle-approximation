# -*- coding: utf-8 -*-
"""Generate publication figures for multi-norm isosceles approximation.

Run ``python generate_figures.py --paper --balance --english --output-dir .``
to reproduce the English figures. Branches I, II, and III use the shortest,
middle, and longest sorted input side as the base, respectively.

The 0.01 value is an input-triangle lattice spacing, not a target-side
discretization. Decision regions use floating-point half-plane clipping of
piecewise-affine costs; tolerances only identify numerical ties and support
geometric clipping. If an angle coefficient rounds to the degenerate limit
2 in float64, use higher-precision arithmetic rather than changing the
requested interval.
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.tri import Triangulation

from verify_multinorm_grid import (
    branch_formula,
    perimeter_one_grid,
    triangular_mesh as multinorm_triangular_mesh,
)


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "pic"
STEP = 0.01
SUM_EDGE = 1.0
SURF_N = 220
EPS = np.finfo(float).eps
TIE_RTOL = 64 * EPS
GEOM_TOL = 128 * EPS
SQRT3_2 = np.sqrt(3.0) / 2.0
PAPER_INTERVALS = [(1.0, 90.0), (45.0, 90.0), (30.0, 50.0), (90.0, 120.0)]
ALPHA_DEG, BETA_DEG = 44.0, 85.0
ALPHA = BETA = K1 = K2 = None
MASK_COLORS = {
    1: "#1f77b4", 2: "#009e73", 4: "#add8e6",
    3: "#7b64b3", 5: "#e8b44d", 6: "#69a78c", 7: "#cc79a7",
}
MASK_LABELS = {
    1: "分支 I", 2: "分支 II", 4: "分支 III",
    3: "I、II 并列", 5: "I、III 并列", 6: "II、III 并列",
    7: "I、II、III 并列",
}
MASK_LABELS_EN = {
    1: "Branch I", 2: "Branch II", 4: "Branch III",
    3: "Branches I and II tie", 5: "Branches I and III tie",
    6: "Branches II and III tie", 7: "All branches tie",
}
MASK_LABELS_RU = {
    1: "Ветвь I", 2: "Ветвь II", 4: "Ветвь III",
    3: "Равенство ветвей I и II", 5: "Равенство ветвей I и III",
    6: "Равенство ветвей II и III", 7: "Равенство всех ветвей",
}

# Publication figures need readable text after being scaled to a manuscript column.
# Adjust these values together rather than changing scattered literal font sizes.
BASE_FONT_SIZE = 12
TERNARY_TICK_FONT_SIZE = 10
TERNARY_AXIS_FONT_SIZE = 13
PANEL_TITLE_FONT_SIZE = 14
SUPTITLE_FONT_SIZE = 15
LEGEND_FONT_SIZE = 13
COLORBAR_LABEL_FONT_SIZE = 13
COLORBAR_TICK_FONT_SIZE = 10


def configure_fonts(language="zh"):
    available = {font.name for font in font_manager.fontManager.ttflist}
    if language == "zh":
        primary = next((name for name in ("SimSun", "Microsoft YaHei", "SimHei")
                        if name in available), "DejaVu Sans")
        fallback = "DejaVu Serif"
    else:
        primary = next((name for name in ("Times New Roman", "DejaVu Serif", "DejaVu Sans")
                        if name in available), "DejaVu Serif")
        fallback = "DejaVu Sans"
    plt.rcParams.update({
        "font.family": [primary, fallback],
        "mathtext.fontset": "stix", "mathtext.default": "regular",
        "axes.unicode_minus": False, "font.size": BASE_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE, "axes.linewidth": 0.9,
        "figure.dpi": 120, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
    })
    return primary


def angle_coefficients(alpha_deg, beta_deg):
    alpha, beta = float(alpha_deg), float(beta_deg)
    if not (math.isfinite(alpha) and math.isfinite(beta)
            and 0 < alpha <= beta < 180):
        raise ValueError("Angles must be finite and satisfy 0 < alpha <= beta < 180.")

    def coefficient(degrees):
        if degrees == 60.0:
            return 1.0
        k = 2.0 * math.sin(degrees * (math.pi / 360.0))
        if k >= 2.0:
            raise FloatingPointError(
                "float64 cannot distinguish {:.17g} degrees from the "
                "degenerate 180-degree limit; use higher-precision "
                "arithmetic without changing the requested interval.".format(degrees))
        if k <= 0:
            raise FloatingPointError(
                "The positive coefficient for this angle is below the float64 range.")
        return k

    return coefficient(alpha), coefficient(beta)


def setup_angles(alpha_deg, beta_deg):
    global ALPHA_DEG, BETA_DEG, ALPHA, BETA, K1, K2
    k1, k2 = angle_coefficients(alpha_deg, beta_deg)
    ALPHA_DEG, BETA_DEG = float(alpha_deg), float(beta_deg)
    ALPHA, BETA = np.deg2rad([ALPHA_DEG, BETA_DEG])
    K1, K2 = k1, k2


def ask_angles():
    def ask(prompt, default, lower, inclusive):
        while True:
            raw = input("{} [default {:g}]: ".format(prompt, default)).strip()
            try:
                value = default if raw == "" else float(raw)
                valid_lower = value >= lower if inclusive else value > lower
                if math.isfinite(value) and valid_lower and value < 180:
                    return value
            except ValueError:
                pass
            print("Enter a finite value that satisfies the angle bounds.")
    alpha = ask("alpha (degrees)", 44.0, 0.0, False)
    beta = ask("beta (degrees)", max(85.0, alpha), alpha, True)
    return alpha, beta


def _coefficients(k1, k2):
    lo = K1 if k1 is None else float(k1)
    hi = K2 if k2 is None else float(k2)
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi < 2):
        raise ValueError("Coefficients must be finite and satisfy 0 < k1 <= k2 < 2.")
    return lo, hi


def _positive_triple(sides):
    values = np.asarray(sides, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Provide three finite, positive side lengths.")
    scale = float(np.max(values))
    normalized = values / scale
    if np.any(normalized == 0):
        raise FloatingPointError(
            "The side-length dynamic range exceeds float64 normalization range.")
    return values, normalized, scale


def _triangle(sides, require_sorted=False):
    values, normalized, scale = _positive_triple(sides)
    order = np.argsort(normalized, kind="stable")
    s = normalized[order]
    if require_sorted and np.any(np.diff(normalized) < 0):
        raise ValueError("The sorted interface requires a <= b <= c.")
    # Use a > c-b to avoid losing a in a+b near degeneracy or at small scales.
    if not s[0] > s[2] - s[1]:
        raise ValueError("The input must be a nondegenerate triangle.")
    return values, s, scale, order


def _branch_cost(s1, s2, s3, k1, k2):
    return (s2 - s1) + max(0.0, s3 - k2 * s2, k1 * s1 - s3)


def unified_cost(s1, s2, s3, k1=None, k2=None):
    k1, k2 = _coefficients(k1, k2)
    _, s, scale = _positive_triple([s1, s2, s3])
    if s[0] > s[1]:
        raise ValueError("unified_cost requires s1 <= s2.")
    return float(_branch_cost(*s, k1, k2) * scale)


def _branch_solution(s1, s2, s3, k1, k2):
    if s3 > k2 * s2:
        return s2, k2 * s2
    if s3 < k1 * s1:
        return s1, k1 * s1
    # Select the upper endpoint of the plateau; avoid division by tiny k1.
    return (s2 if s3 >= k1 * s2 else s3 / k1), s3


def _costs(s, k1, k2):
    a, b, c = s
    return np.array([_branch_cost(b, c, a, k1, k2),
                     _branch_cost(a, c, b, k1, k2),
                     _branch_cost(a, b, c, k1, k2)])


def optimal_mask(costs):
    costs = np.asarray(costs, dtype=float)
    tolerance = TIE_RTOL * max(1.0, float(np.max(np.abs(costs))))
    return sum(1 << i for i, c in enumerate(costs) if c <= np.min(costs) + tolerance)


def _scaled_output(cost, l, m, scale, k1, k2):
    result = np.array([cost, l, m]) * scale
    # Rescaling l and m separately adds rounding; recompute bounds at output scale.
    # This matters when k2=nextafter(2,0), where a valid m can round to 2*l.
    lower, upper = k1 * result[1], k2 * result[1]
    result[2] = min(upper, max(lower, result[2]))
    if not np.all(np.isfinite(result)) or np.any(result[1:] <= 0):
        raise FloatingPointError("The optimal output exceeds the float64 range.")
    if not result[2] / 2 < result[1]:
        raise FloatingPointError(
            "The strict triangle inequality cannot be represented after rescaling.")
    return float(result[0]), (float(result[1]), float(result[2]))


def solve_triangle(sides, k1=None, k2=None):
    """Return a solution preserving input-side indices.

    ``sort_order[i]`` is the original index of the ith sorted side.
    ``selected_branch`` is I or III, selecting I for exactly equal float costs.
    ``optimal_branches`` includes every branch within ``TIE_RTOL`` and may
    include II.
    """
    k1, k2 = _coefficients(k1, k2)
    values, s, scale, order = _triangle(sides)
    costs = _costs(s, k1, k2)
    branch = 1 if costs[0] <= costs[2] else 3
    base_index = branch - 1
    legs = [s[i] for i in range(3) if i != base_index]
    l, m = _branch_solution(*legs, s[base_index], k1, k2)
    cost, (l_out, m_out) = _scaled_output(costs[base_index], l, m, scale, k1, k2)
    target_sorted = np.full(3, l_out)
    target_sorted[base_index] = m_out
    target = np.empty(3)
    target[order] = target_sorted
    mask = optimal_mask(costs)
    return {
        "cost": cost, "selected_branch": branch,
        "optimal_branches": [i + 1 for i in range(3) if mask & (1 << i)],
        "optimal_mask": mask, "l": l_out, "m": m_out,
        "target_sides": target.tolist(), "sort_order": order.tolist(),
        "branch_costs": (costs * scale).tolist(),
        "diff": float((costs[0] - costs[2]) * scale),
        "tie_relative_tolerance": TIE_RTOL,
    }


def analytic_solve_sorted(a, b, c, k1=None, k2=None):
    _triangle([a, b, c], require_sorted=True)
    return analytic_solve_any(a, b, c, k1, k2)


def analytic_solve_any(a, b, c, k1=None, k2=None):
    result = solve_triangle([a, b, c], k1, k2)
    return (result["cost"], result["selected_branch"],
            (result["l"], result["m"]), result["diff"])


def _feasible(l, m, k1, k2):
    if not (math.isfinite(l) and math.isfinite(m) and l > 0 and m > 0 and m / 2 < l):
        return False
    tolerance = TIE_RTOL * max(l, m)
    return k1 * l - tolerance <= m <= k2 * l + tolerance


def _best_for_branch(s1, s2, s3, k1, k2):
    """Independently enumerate piecewise-linear breakpoints without formulas."""
    candidates = [(l, s3) for l in (s1, s2) if k1 * l <= s3 <= k2 * l]
    for k in (k1, k2):
        candidates.extend([(s1, k * s1), (s2, k * s2)])
        # Outside [s1,s2], leg-residual improvement has rate 2, exceeding
        # the worst base-residual change rate k<2, so no further crossings matter.
        if k * s1 <= s3 <= k * s2:
            candidates.append((s3 / k, s3))
    evaluated = [(abs(l - s1) + abs(l - s2) + abs(m - s3), l, m)
                 for l, m in candidates if _feasible(l, m, k1, k2)]
    if not evaluated:
        raise FloatingPointError("No representable positive, finite candidate exists.")
    return min(evaluated, key=lambda item: item[0])


def enumerate_details(sides, k1=None, k2=None):
    k1, k2 = _coefficients(k1, k2)
    _, s, scale, _ = _triangle(sides)
    a, b, c = s
    answers = [_best_for_branch(*mapping, k1, k2)
               for mapping in ((b, c, a), (a, c, b), (a, b, c))]
    costs = [answer[0] for answer in answers]
    bid = int(np.argmin(costs))
    cost, lm = _scaled_output(*answers[bid], scale, k1, k2)
    return {"cost": cost, "selected_branch": bid + 1, "lm": lm,
            "optimal_mask": optimal_mask(costs),
            "branch_costs": (np.array(costs) * scale).tolist()}


def enumerate_solve_sorted(a, b, c, k1=None, k2=None):
    _triangle([a, b, c], require_sorted=True)
    return enumerate_solve_any(a, b, c, k1, k2)


def enumerate_solve_any(a, b, c, k1=None, k2=None):
    result = enumerate_details([a, b, c], k1, k2)
    return result["cost"], result["selected_branch"], result["lm"]


def generate_triangle_grid(n=SURF_N, inset=1e-4):
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError("n must be a positive integer.")
    if not (0 < inset < 1):
        raise ValueError("The centroid inset fraction must lie in (0, 1).")
    uv = np.array([(i / n, j / n) for i in range(n + 1)
                   for j in range(n + 1 - i)])
    u, v = uv.T
    closed = np.column_stack([0.5 * (1 - v), 0.5 * (u + v), 0.5 * (1 - u)])
    data = (1 - inset) * closed + inset / 3
    if not (np.all(data > 0) and np.all(data < 0.5)):
        raise FloatingPointError(
            "The inset is too small: the float64 grid remains on the degenerate boundary.")
    return tuple(data.T)


def generate_lattice_points(step=STEP, total=SUM_EDGE):
    if not (math.isfinite(step) and math.isfinite(total) and 0 < step < total):
        raise ValueError("step and total must be finite and satisfy 0 < step < total.")
    ratio = total / step
    n = int(round(ratio))
    if abs(ratio - n) > TIE_RTOL * ratio:
        raise ValueError("total must be an integer multiple of the input-lattice step.")
    pts = [(i, j, n - i - j) for i in range(1, n)
           for j in range(1, n - i)
           if 2 * max(i, j, n - i - j) < n]
    if not pts:
        raise ValueError("The input lattice is too coarse to contain nondegenerate triangles.")
    return (np.asarray(pts, dtype=float) * (total / n)).tolist()


def tern_cart(a, b, c):
    return np.asarray(b) + 0.5 * np.asarray(c), SQRT3_2 * np.asarray(c)


def _draw_ternary_frame(ax):
    for t in np.arange(0.1, 1, 0.1):
        ax.plot([1 - t, 0.5 * (1 - t)], [0, SQRT3_2 * (1 - t)],
                color="0.86", lw=0.35, zorder=1)
        ax.plot([t, t + 0.5 * (1 - t)], [0, SQRT3_2 * (1 - t)],
                color="0.86", lw=0.35, zorder=1)
        ax.plot([0.5 * t, 1 - 0.5 * t], [SQRT3_2 * t] * 2,
                color="0.86", lw=0.35, zorder=1)
        ax.text(t, -0.028, "{:.1f}".format(t), ha="center", va="top",
                fontsize=TERNARY_TICK_FONT_SIZE)
        ax.text(1 - 0.5 * t + 0.025, SQRT3_2 * t, "{:.1f}".format(t),
                ha="left", va="center", fontsize=TERNARY_TICK_FONT_SIZE)
        ax.text(0.5 * (1 - t) - 0.025, SQRT3_2 * (1 - t), "{:.1f}".format(t),
                ha="right", va="center", fontsize=TERNARY_TICK_FONT_SIZE)
    ax.plot([0, 1, 0.5, 0], [0, 0, SQRT3_2, 0], color="black", lw=0.9)
    valid = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    x, y = tern_cart(*valid.T)
    ax.plot(np.r_[x, x[0]], np.r_[y, y[0]], color="0.35", ls="--", lw=0.7, zorder=10)
    ax.text(0.5, -0.095, "$b$", ha="center", fontsize=TERNARY_AXIS_FONT_SIZE)
    ax.text(0.14, 0.47, "$a$", ha="center", fontsize=TERNARY_AXIS_FONT_SIZE)
    ax.text(0.86, 0.47, "$c$", ha="center", fontsize=TERNARY_AXIS_FONT_SIZE)
    ax.set(xlim=(-0.10, 1.10), ylim=(-0.135, 0.97), aspect="equal")
    ax.axis("off")


def _save_figure(fig, save_path):
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(str(path.with_suffix(suffix)), bbox_inches="tight")
    plt.close(fig)
    print("saved: {}".format(path.with_suffix(".png")))


def _clip_polygon(poly, coefficients):
    if len(poly) == 0:
        return poly
    output = []
    prev, fprev = poly[-1], float(np.dot(coefficients, poly[-1]))
    for curr in poly:
        fcurr = float(np.dot(coefficients, curr))
        inside_prev, inside_curr = fprev >= -GEOM_TOL, fcurr >= -GEOM_TOL
        if inside_curr != inside_prev:
            weight = np.clip(fprev / (fprev - fcurr), 0, 1)
            output.append(prev + weight * (curr - prev))
        if inside_curr:
            output.append(curr)
        prev, fprev = curr, fcurr
    return np.asarray(output)


def _polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    x, y = tern_cart(*np.asarray(poly).T)
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _zero_segment(poly, coeff):
    points = []
    for p, q in zip(poly, np.roll(poly, -1, axis=0)):
        fp, fq = np.dot(coeff, p), np.dot(coeff, q)
        if abs(fp) <= GEOM_TOL:
            points.append(p)
        if fp * fq < 0:
            points.append(p + fp / (fp - fq) * (q - p))
    if len(points) < 2:
        return None
    p, q = max(itertools.combinations(points, 2), key=lambda pq: np.linalg.norm(pq[0] - pq[1]))
    return np.array([p, q]) if np.linalg.norm(p - q) > GEOM_TOL else None


def decision_partition(k1, k2):
    """Partition a sorted sector by active max terms and clip J1-J3 half-planes.

    Return floating-point polygons and existing tie segments without collapsing
    two-dimensional ties to arbitrary colors. In the balanced case,
    ``{a <= k1*b, c >= k2*b}`` must still meet the nondegenerate domain; no
    two-dimensional cell exists when k1 <= 1/2.
    """
    k1, k2 = _coefficients(k1, k2)
    wedge = np.array([[0, 0.5, 0.5], [0.25, 0.25, 0.5], [1/3, 1/3, 1/3]])
    p1 = np.array([[0, 0, 0], [1, 0, -k2], [-1, k1, 0]])
    p3 = np.array([[0, 0, 0], [0, -k2, 1], [k1, 0, -1]])
    base1, base3 = np.array([0, -1, 1]), np.array([-1, 1, 0])
    cells, segments = [], []
    for i, j in itertools.product(range(3), repeat=2):
        poly = wedge.copy()
        for h in range(3):
            poly = _clip_polygon(poly, p1[i] - p1[h])
            poly = _clip_polygon(poly, p3[j] - p3[h])
        if _polygon_area(poly) <= GEOM_TOL ** 2:
            continue
        delta = base1 + p1[i] - base3 - p3[j]
        if np.max(np.abs(delta)) <= GEOM_TOL:
            cells.append((poly, optimal_mask(_costs(poly.mean(axis=0), k1, k2))))
            continue
        for sign in (1, -1):
            half = _clip_polygon(poly, sign * delta)
            if _polygon_area(half) > GEOM_TOL ** 2:
                cells.append((half, optimal_mask(_costs(half.mean(axis=0), k1, k2))))
        segment = _zero_segment(poly, delta)
        if segment is not None:
            a, b, c = segment.mean(axis=0)
            pen1 = max(0, a - k2 * c, k1 * b - a)
            pen3 = max(0, c - k2 * b, k1 * a - c)
            segments.append((segment, "core" if max(pen1, pen3) <= GEOM_TOL else "outer"))
    return cells, segments


def partition_summary(k1, k2):
    cells, segments = decision_partition(k1, k2)
    area = sum(_polygon_area(poly) for poly, _ in cells)
    tie_area = sum(_polygon_area(poly) for poly, mask in cells if mask not in (1, 2, 4))
    return {"sorted_cell_count": len(cells), "sorted_total_area": area,
            "tie_area_fraction": tie_area / area,
            "boundary_types": sorted({kind for _, kind in segments}),
            "geometry_tolerance": GEOM_TOL}


def outer_tie_annotation(k1, k2, language="zh"):
    total = k1 + k2
    if total < 2.0 - GEOM_TOL:
        return (r"$\min(a,b,c)=(2-k_2)\,\mathrm{M}(a,b,c)$")
    if total > 2.0 + GEOM_TOL:
        return (r"$\max(a,b,c)=(2-k_1)\,\mathrm{M}(a,b,c)$")
    connector = {"zh": " 或 ", "en": r" or ", "ru": r" или "}[language]
    return (r"$\min(a,b,c)=k_1\,\mathrm{M}(a,b,c)$" + connector +
            r"$\max(a,b,c)=k_2\,\mathrm{M}(a,b,c)$")


def plot_combined_decisions(data_e, mask_e, k1, k2, save_path, title,
                            language="zh", show_tie_segments=True):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
    cells, segments = decision_partition(k1, k2)
    labels = {"zh": MASK_LABELS, "en": MASK_LABELS_EN, "ru": MASK_LABELS_RU}[language]
    displayed = set()
    centroid = np.full(3, 1 / 3)
    repeated_ties = []
    whole_domain_tie = all(mask == 7 for _, mask in cells)
    if show_tie_segments and not whole_domain_tie:
        for endpoint in ([0.25, 0.25, 0.5], [0, 0.5, 0.5]):
            segment = np.array([endpoint, centroid])
            mask = optimal_mask(_costs(segment.mean(axis=0), k1, k2))
            if mask & 2 and mask not in (1, 2, 4):
                repeated_ties.append((segment, mask))
    for permutation in itertools.permutations(range(3)):
        for poly, mask in cells:
            displayed.add(mask)
            x, y = tern_cart(*poly[:, permutation].T)
            axes[0].add_patch(Polygon(np.column_stack([x, y]), closed=True,
                                     facecolor=MASK_COLORS[mask], edgecolor="none",
                                     alpha=0.88, zorder=2))
        if show_tie_segments:
            for segment, kind in segments:
                x, y = tern_cart(*segment[:, permutation].T)
                axes[0].plot(x, y, color="#404040" if kind == "core" else "#777777",
                             ls="-" if kind == "core" else "--", lw=1.5, zorder=5)
            for segment, mask in repeated_ties:
                displayed.add(mask)
                x, y = tern_cart(*segment[:, permutation].T)
                axes[0].plot(x, y, color=MASK_COLORS[mask], lw=1.4, zorder=6)
    if not whole_domain_tie:
        displayed.add(7)
        x, y = tern_cart(*centroid)
        axes[0].scatter([x], [y], color=MASK_COLORS[7], s=13, zorder=7)
    x, y = tern_cart(*data_e.T)
    axes[1].scatter(x, y, c=[MASK_COLORS[int(mask)] for mask in mask_e],
                    s=4, edgecolors="none", rasterized=True, zorder=2)
    displayed.update(int(mask) for mask in mask_e)
    headings = {
        "zh": ("(a) 解析分片区域", "(b) 有限候选枚举：输入格点"),
        "en": ("(a) Analytic decision regions",
               "(b) Finite-candidate enumeration: input lattice"),
        "ru": ("(a) Аналитические области выбора",
               "(b) Перебор конечных кандидатов: сетка входов"),
    }[language]
    for ax, heading in zip(axes, headings):
        _draw_ternary_frame(ax)
        ax.set_title(heading, fontsize=PANEL_TITLE_FONT_SIZE, pad=8)
    handles = []
    for mask in sorted(displayed):
        if mask == 7 and not whole_domain_tie and not any(m == 7 for _, m in repeated_ties):
            handles.append(Line2D([0], [0], color=MASK_COLORS[mask], marker="o",
                                  ls="none", markersize=7,
                                  label={
                                      "zh": "等边输入：三分支并列",
                                      "en": "Equilateral input: all branches tie",
                                      "ru": "Равносторонний вход: равенство всех ветвей",
                                  }[language]))
        else:
            handles.append(Patch(facecolor=MASK_COLORS[mask], label=labels[mask]))
    if show_tie_segments:
        kinds = {kind for _, kind in segments}
        if "core" in kinds:
            handles.append(Line2D([0], [0], color="#404040", lw=2.0,
                                  label={
                                      "zh": r"核心等代价线：$\mathrm{M}(a,b,c)=1/3$",
                                      "en": r"Core tie line: $\mathrm{M}(a,b,c)=1/3$",
                                      "ru": r"Основная линия равной стоимости: $\mathrm{M}(a,b,c)=1/3$",
                                  }[language]))
        if "outer" in kinds:
            handles.append(Line2D([0], [0], color="#777777", ls="--", lw=2.0,
                                  label={
                                      "zh": "其余等代价线段",
                                      "en": "Other tie segments",
                                      "ru": "Другие отрезки равной стоимости",
                                  }[language]))
            handles.append(Line2D([0], [0], color="none", lw=0,
                                  label=outer_tie_annotation(k1, k2, language)))
    fig.suptitle(title + {
        "zh": "；分支按输入边长排序定义",
        "en": "; branches are defined after sorting the input sides",
        "ru": "; ветви определяются после упорядочения входных сторон",
    }[language],
                 fontsize=SUPTITLE_FONT_SIZE, y=0.985)
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.035), frameon=False,
               fontsize=LEGEND_FONT_SIZE, handlelength=2.1,
               columnspacing=1.8, labelspacing=0.7)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    _save_figure(fig, save_path)


def plot_combined_costs(data_s, cost_s, data_e, cost_e, save_path, title, language="zh"):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
    vmin = min(float(np.min(cost_s)), float(np.min(cost_e)))
    vmax = max(float(np.max(cost_s)), float(np.max(cost_e)))
    if not vmax > vmin:
        vmax = np.nextafter(vmin, math.inf)
    norm = Normalize(vmin=vmin, vmax=vmax)
    x, y = tern_cart(*data_s.T)
    axes[0].tricontourf(Triangulation(x, y), cost_s, levels=np.linspace(vmin, vmax, 25),
                        cmap="viridis", norm=norm, zorder=2)
    x, y = tern_cart(*data_e.T)
    axes[1].scatter(x, y, c=cost_e, s=4, cmap="viridis", norm=norm,
                    edgecolors="none", rasterized=True, zorder=2)
    headings = {
        "zh": ("(a) 解析代价：网格插值", "(b) 有限候选枚举代价"),
        "en": ("(a) Analytic cost: interpolated grid",
               "(b) Finite-candidate enumeration cost"),
        "ru": ("(a) Аналитическая стоимость: интерполированная сетка",
               "(b) Стоимость при переборе конечных кандидатов"),
    }[language]
    for ax, heading in zip(axes, headings):
        _draw_ternary_frame(ax)
        ax.set_title(heading, fontsize=PANEL_TITLE_FONT_SIZE, pad=8)
    fig.suptitle(title + {
        "zh": "；左右共用同一色标",
        "en": "; shared color scale",
        "ru": "; общая цветовая шкала",
    }[language],
                 fontsize=SUPTITLE_FONT_SIZE, y=0.985)
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.04, top=0.89, wspace=0.12)
    colorbar_ax = fig.add_axes([0.91, 0.16, 0.020, 0.63])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), cax=colorbar_ax)
    colorbar.set_label({
        "zh": "最优 L1 代价",
        "en": "Optimal L1 cost",
        "ru": "Оптимальная L1-стоимость",
    }[language],
                       fontsize=COLORBAR_LABEL_FONT_SIZE, labelpad=10)
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE)
    _save_figure(fig, save_path)


def _multinorm_global_answer(sides, k1, k2, p):
    """Return the global cost and branch mask after sorting the input sides."""
    a, b, c = sorted(sides)
    values = np.array([
        branch_formula(b, c, a, k1, k2, p)[0],
        branch_formula(a, c, b, k1, k2, p)[0],
        branch_formula(a, b, c, k1, k2, p)[0],
    ])
    best = float(np.min(values))
    tolerance = TIE_RTOL * max(1.0, abs(best))
    mask = sum(1 << index for index, value in enumerate(values)
               if value <= best + tolerance)
    return best, mask


def plot_multinorm_figures():
    """Generate \ell_2 and \ell_infty figures in the \ell_1 publication format."""
    spec = interval_specs()[0]
    k1, k2 = spec["k1"], spec["k2"]
    data_e = np.asarray(perimeter_one_grid(), dtype=float)
    data_s = multinorm_triangular_mesh(SURF_N)
    title = r"$[\alpha,\beta]=[1^\circ,90^\circ]$"

    for p, stem, norm_name in (
            (2, "l2", r"\ell_2"),
            (math.inf, "linf", r"\ell_\infty")):
        dense_answers = [_multinorm_global_answer(row, k1, k2, p)
                         for row in data_s]
        exact_answers = [_multinorm_global_answer(row, k1, k2, p)
                         for row in data_e]
        cost_s = np.array([answer[0] for answer in dense_answers])
        cost_e = np.array([answer[0] for answer in exact_answers])
        mask_s = np.array([answer[1] for answer in dense_answers])
        mask_e = np.array([answer[1] for answer in exact_answers])
        displayed = sorted(set(mask_s) | set(mask_e))

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
        palette = ListedColormap([MASK_COLORS[mask] for mask in displayed])
        norm = BoundaryNorm(np.arange(-0.5, len(displayed) + 0.5),
                            len(displayed))
        code_by_mask = {mask: index for index, mask in enumerate(displayed)}
        x, y = tern_cart(*data_s.T)
        triangulation = Triangulation(x, y)
        face_codes = np.array([
            code_by_mask[mask_s[triangle[0]]]
            for triangle in triangulation.triangles
        ])
        axes[0].tripcolor(triangulation, facecolors=face_codes, cmap=palette,
                          norm=norm, edgecolors="none", zorder=2)
        x, y = tern_cart(*data_e.T)
        axes[1].scatter(x, y,
                        c=[MASK_COLORS[int(mask)] for mask in mask_e],
                        s=4, edgecolors="none", rasterized=True, zorder=2)
        headings = (
            "(a) Explicit candidate-rule regions",
            "(b) Explicit candidate rule: input lattice",
        )
        for ax, heading in zip(axes, headings):
            _draw_ternary_frame(ax)
            ax.set_title(heading, fontsize=PANEL_TITLE_FONT_SIZE, pad=8)
        handles = [Patch(facecolor=MASK_COLORS[mask],
                         label=MASK_LABELS_EN[mask]) for mask in displayed]
        fig.suptitle(title + r"; optimal {} branches are defined after sorting the input sides".format(norm_name),
                     fontsize=SUPTITLE_FONT_SIZE, y=0.985)
        fig.legend(handles=handles, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, -0.035), frameon=False,
                   fontsize=LEGEND_FONT_SIZE, handlelength=2.1,
                   columnspacing=1.8, labelspacing=0.7)
        fig.tight_layout(rect=(0, 0.10, 1, 0.94))
        _save_figure(fig, ROOT / ("1-90_{}_decision_regions.png".format(stem)))

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
        vmin = min(float(np.min(cost_s)), float(np.min(cost_e)))
        vmax = max(float(np.max(cost_s)), float(np.max(cost_e)))
        if not vmax > vmin:
            vmax = np.nextafter(vmin, math.inf)
        norm = Normalize(vmin=vmin, vmax=vmax)
        x, y = tern_cart(*data_s.T)
        axes[0].tricontourf(Triangulation(x, y), cost_s,
                            levels=np.linspace(vmin, vmax, 25),
                            cmap="viridis", norm=norm, zorder=2)
        x, y = tern_cart(*data_e.T)
        axes[1].scatter(x, y, c=cost_e, s=4, cmap="viridis", norm=norm,
                        edgecolors="none", rasterized=True, zorder=2)
        headings = (
            "(a) Explicit candidate rule: interpolated grid",
            "(b) Explicit candidate rule: input lattice",
        )
        for ax, heading in zip(axes, headings):
            _draw_ternary_frame(ax)
            ax.set_title(heading, fontsize=PANEL_TITLE_FONT_SIZE, pad=8)
        fig.suptitle(title + r"; optimal {} cost, shared color scale".format(norm_name),
                     fontsize=SUPTITLE_FONT_SIZE, y=0.985)
        fig.subplots_adjust(left=0.02, right=0.88, bottom=0.04, top=0.89,
                            wspace=0.12)
        colorbar_ax = fig.add_axes([0.91, 0.16, 0.020, 0.63])
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"),
                                cax=colorbar_ax)
        colorbar.set_label(r"Optimal {} cost".format(norm_name),
                           fontsize=COLORBAR_LABEL_FONT_SIZE, labelpad=10)
        colorbar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE)
        _save_figure(fig, ROOT / ("1-90_{}_costs.png".format(stem)))


def interval_specs(include_balance=False):
    specs = [{"name": "{:g}-{:g}".format(a, b), "alpha_deg": a, "beta_deg": b,
              "k1": angle_coefficients(a, b)[0], "k2": angle_coefficients(a, b)[1]}
             for a, b in PAPER_INTERVALS]
    if include_balance:
        specs.append({"name": "balance", "alpha_deg": math.degrees(2 * math.asin(0.75 / 2)),
                      "beta_deg": math.degrees(2 * math.asin(1.25 / 2)),
                      "k1": 0.75, "k2": 1.25})
    return specs


def generate_figures(spec, step=STEP, surf_n=SURF_N, output_dir=FIG_DIR,
                     language="zh", show_tie_segments=True, output_stem=None):
    k1, k2 = spec["k1"], spec["k2"]
    data_e = np.asarray(generate_lattice_points(step))
    analytic = [solve_triangle(row, k1, k2) for row in data_e]
    enum = [enumerate_details(row, k1, k2) for row in data_e]
    cost_a = np.array([answer["cost"] for answer in analytic])
    cost_e = np.array([answer["cost"] for answer in enum])
    masks = np.array([answer["optimal_mask"] for answer in enum])
    data_s = np.column_stack(generate_triangle_grid(surf_n))
    sorted_s = np.sort(data_s, axis=1)
    a, b, c = sorted_s.T
    j1 = c - b + np.maximum.reduce([np.zeros_like(a), a - k2 * c, k1 * b - a])
    j3 = b - a + np.maximum.reduce([np.zeros_like(a), c - k2 * b, k1 * a - c])
    title = r"$[\alpha,\beta]=[{:.4g}^\circ,{:.4g}^\circ]$".format(
        spec["alpha_deg"], spec["beta_deg"])
    if spec["name"] == "balance":
        title = r"$k_1=0.75,\ k_2=1.25,\ k_1+k_2=2$"
    out = Path(output_dir)
    stem = spec["name"] if output_stem is None else output_stem
    plot_combined_decisions(data_e, masks, k1, k2,
                            out / (stem + "_decision_regions.png"), title,
                            language=language, show_tie_segments=show_tie_segments)
    plot_combined_costs(data_s, np.minimum(j1, j3), data_e, cost_e,
                        out / (stem + "_costs.png"), title, language=language)
    return dict(spec, input_grid_spacing=step, input_grid_count=len(data_e),
                surface_grid_count=len(data_s),
                max_cost_diff=float(np.max(np.abs(cost_a - cost_e))),
                mean_cost_diff=float(np.mean(np.abs(cost_a - cost_e))),
                tie_count=int(sum(mask not in (1, 2, 4) for mask in masks)),
                partition=partition_summary(k1, k2))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper", action="store_true",
                        help="Generate the four standard angle-interval figure sets (default).")
    parser.add_argument("--balance", action="store_true",
                        help="Also generate the balanced-coefficient tie-region figures.")
    parser.add_argument("--alpha", type=float,
                        help="Lower angle in degrees; must be supplied with --beta.")
    parser.add_argument("--beta", type=float,
                        help="Upper angle in degrees; must be supplied with --alpha.")
    parser.add_argument("--interactive", action="store_true",
                        help="Prompt for alpha and beta in degrees.")
    parser.add_argument("--step", type=float, default=STEP,
                        help="Input-triangle lattice spacing, not a target-side step.")
    parser.add_argument("--surf-n", type=int, default=SURF_N,
                        help="Number of subdivisions per side of the surface mesh.")
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR,
                        help="Directory for L1 figure pairs and figure_generation_results.json.")
    parser.add_argument("--english", action="store_true",
                        help="Use English titles, legends, and colorbar labels.")
    parser.add_argument("--russian", action="store_true",
                        help="Use Russian titles, legends, and colorbar labels.")
    parser.add_argument("--hide-tie-lines", action="store_true",
                        help="Hide tie segments and omit them from the legend.")
    parser.add_argument("--output-stem", type=str,
                        help="Prefix for every L1 output filename.")
    parser.add_argument("--no-results-json", action="store_true",
                        help="Do not write figure_generation_results.json.")
    args = parser.parse_args()
    if args.english and args.russian:
        parser.error("--english and --russian cannot be used together")
    language = "ru" if args.russian else "en" if args.english else "zh"
    configure_fonts(language)
    if args.interactive:
        args.alpha, args.beta = ask_angles()
    if (args.alpha is None) != (args.beta is None):
        parser.error("--alpha and --beta must be supplied together")
    if args.alpha is not None:
        if args.paper:
            parser.error("--paper cannot be combined with an explicit angle interval")
        k1, k2 = angle_coefficients(args.alpha, args.beta)
        specs = [dict(name="{:g}-{:g}".format(args.alpha, args.beta),
                      alpha_deg=args.alpha, beta_deg=args.beta, k1=k1, k2=k2)]
        if args.balance:
            specs.append(interval_specs(True)[-1])
    else:
        specs = interval_specs(args.balance)
    results = [generate_figures(spec, args.step, args.surf_n, args.output_dir,
                                language=language,
                                show_tie_segments=not args.hide_tie_lines,
                                output_stem=args.output_stem) for spec in specs]
    if language == "en" and args.paper:
        plot_multinorm_figures()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_results_json:
        with (args.output_dir / "figure_generation_results.json").open("w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(results, ensure_ascii=False, indent=2))


setup_angles(ALPHA_DEG, BETA_DEG)

if __name__ == "__main__":
    main()
