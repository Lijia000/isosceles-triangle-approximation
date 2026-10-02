"""Reproducible perimeter-one grid verification for all three norms.

Run ``python verify_multinorm_grid.py`` in this directory.  The program uses
the exact 0.01 input grid, preserving the three input positions, and writes
verification_results.json and verification_summary.tex next to this file.
"""

import json
import math
import platform
import sys
from pathlib import Path

import matplotlib
import numpy as np
import scipy
from scipy.optimize import linprog, minimize_scalar


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation


ROOT = Path(__file__).resolve().parent
GRID_STEP = 0.01
TOL = 2.0e-7
TIE_TOL = 64.0 * np.finfo(float).eps
SQRT3_2 = math.sqrt(3.0) / 2.0
BRANCH_COLORS = {
    "I": "#0072B2",
    "II": "#009E73",
    "III": "#D55E00",
    "Tie": "#CC79A7",
}


def settings():
    return (
        ("[1,90] degrees", 2 * math.sin(math.radians(1) / 2),
         2 * math.sin(math.radians(90) / 2)),
        ("[45,90] degrees", 2 * math.sin(math.radians(45) / 2),
         2 * math.sin(math.radians(90) / 2)),
        ("[30,50] degrees", 2 * math.sin(math.radians(30) / 2),
         2 * math.sin(math.radians(50) / 2)),
        ("[90,120] degrees", 2 * math.sin(math.radians(90) / 2),
         2 * math.sin(math.radians(120) / 2)),
        ("(k1,k2)=(3/4,5/4)", 0.75, 1.25),
    )


def perimeter_one_grid():
    """All ordered, position-preserving .01 triples with strict inequalities."""
    triples = []
    for ai in range(1, 99):
        for bi in range(1, 100 - ai):
            ci = 100 - ai - bi
            if ci > 0 and max(ai, bi, ci) < 50:
                triples.append((ai / 100.0, bi / 100.0, ci / 100.0))
    return triples


def project_base(length, base_input, k1, k2):
    return min(k2 * length, max(base_input, k1 * length))


def norm_value(length, base, s1, s2, s3, p):
    residuals = (length - s1, length - s2, base - s3)
    if p == 1:
        return sum(abs(x) for x in residuals)
    if p == 2:
        return math.sqrt(sum(x * x for x in residuals))
    return max(abs(x) for x in residuals)


def l1_formula(s1, s2, s3, k1, k2):
    lower, upper = sorted((s1, s2))
    if s3 > k2 * upper:
        length, base = upper, k2 * upper
    elif s3 < k1 * lower:
        length, base = lower, k1 * lower
    else:
        length, base = max(lower, s3 / k2), s3
    return norm_value(length, base, s1, s2, s3, 1), length, base


def l2_formula(s1, s2, s3, k1, k2):
    left, right = s3 / k2, s3 / k1
    candidates = (
        min(left, (s1 + s2 + k2 * s3) / (2 + k2 * k2)),
        min(right, max(left, (s1 + s2) / 2)),
        max(right, (s1 + s2 + k1 * s3) / (2 + k1 * k1)),
    )
    values = []
    for length in candidates:
        base = project_base(length, s3, k1, k2)
        values.append((norm_value(length, base, s1, s2, s3, 2), length, base))
    return min(values, key=lambda value: (value[0], value[1]))


def affine_candidates(s1, s2, s3, k1, k2):
    """All endpoints and pairwise signed affine intersections by projection zone."""
    bounds = ((0.0, s3 / k2, k2), (s3 / k2, s3 / k1, 0.0),
              (s3 / k1, math.inf, k1))
    candidates = []
    for lo, hi, slope in bounds:
        if math.isfinite(lo):
            candidates.append(lo)
        if math.isfinite(hi):
            candidates.append(hi)
        lines = ((1.0, -s1), (1.0, -s2),
                 (slope, -s3) if slope else (0.0, 0.0))
        for i in range(3):
            for j in range(i + 1, 3):
                for sign in (-1.0, 1.0):
                    denominator = lines[i][0] - sign * lines[j][0]
                    numerator = sign * lines[j][1] - lines[i][1]
                    if abs(denominator) > 1e-14:
                        length = numerator / denominator
                        if length >= lo - 1e-12 and length <= hi + 1e-12:
                            candidates.append(max(lo, min(length, hi)))
    return candidates


def linf_formula(s1, s2, s3, k1, k2):
    values = []
    for length in affine_candidates(s1, s2, s3, k1, k2):
        if length < 0:
            continue
        base = project_base(length, s3, k1, k2)
        values.append((norm_value(length, base, s1, s2, s3, math.inf),
                       length, base))
    return min(values, key=lambda value: (value[0], value[1]))


def branch_formula(s1, s2, s3, k1, k2, p):
    if p == 1:
        return l1_formula(s1, s2, s3, k1, k2)
    if p == 2:
        return l2_formula(s1, s2, s3, k1, k2)
    return linf_formula(s1, s2, s3, k1, k2)


def l1_breakpoint_enumeration(s1, s2, s3, k1, k2):
    candidates = (s1, s2, s3 / k2, s3 / k1)
    values = []
    for length in candidates:
        base = project_base(length, s3, k1, k2)
        values.append((norm_value(length, base, s1, s2, s3, 1), length, base))
    return min(values, key=lambda value: (value[0], value[1]))


def l1_lp(s1, s2, s3, k1, k2):
    # Variables are (l,m,r1,r2,r3), minimizing r1+r2+r3.
    objective = (0, 0, 1, 1, 1)
    matrix = (
        (1, 0, -1, 0, 0), (-1, 0, -1, 0, 0),
        (1, 0, 0, -1, 0), (-1, 0, 0, -1, 0),
        (0, 1, 0, 0, -1), (0, -1, 0, 0, -1),
        (k1, -1, 0, 0, 0), (-k2, 1, 0, 0, 0),
    )
    rhs = (s1, -s1, s2, -s2, s3, -s3, 0, 0)
    result = linprog(objective, A_ub=np.asarray(matrix, dtype=float),
                     b_ub=np.asarray(rhs, dtype=float), bounds=(0, None),
                     method="revised simplex",
                     options={"tol": 1e-10})
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.fun), float(result.x[0]), float(result.x[1])


def linf_lp(s1, s2, s3, k1, k2):
    # Variables are (l,m,t), minimizing the residual epigraph t.
    objective = (0, 0, 1)
    matrix = (
        (1, 0, -1), (-1, 0, -1),
        (1, 0, -1), (-1, 0, -1),
        (0, 1, -1), (0, -1, -1),
        (k1, -1, 0), (-k2, 1, 0),
    )
    rhs = (s1, -s1, s2, -s2, s3, -s3, 0, 0)
    result = linprog(objective, A_ub=np.asarray(matrix, dtype=float),
                     b_ub=np.asarray(rhs, dtype=float), bounds=(0, None),
                     method="revised simplex",
                     options={"tol": 1e-10})
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.fun), float(result.x[0]), float(result.x[1])


def l2_continuous_reference(s1, s2, s3, k1, k2):
    """Independent direct constrained minimization after base projection."""
    def squared_objective(length):
        base = project_base(length, s3, k1, k2)
        return ((length - s1) ** 2 + (length - s2) ** 2
                + (base - s3) ** 2)

    upper = 4.0 * max(s1, s2, s3 / k1, s3 / k2)
    result = minimize_scalar(squared_objective, bounds=(0.0, upper),
                             method="bounded",
                             options={"xatol": 1e-13, "maxiter": 1000})
    if not result.success:
        raise RuntimeError(result.message)
    length = float(result.x)
    return math.sqrt(float(result.fun)), length, project_base(length, s3, k1, k2)


def branches(triple):
    a, b, c = triple
    return ((b, c, a), (a, c, b), (a, b, c))


def two_kept_branch_indices(triple):
    order = sorted(range(3), key=lambda i: triple[i])
    # The branches are indexed by the original position used as the base.
    return order[0], order[2]


def feasibility_violation(length, base, k1, k2):
    return max(0.0, k1 * length - base, base - k2 * length, -length, -base)


def reconstruction_error(value, length, base, s1, s2, s3, p):
    return abs(value - norm_value(length, base, s1, s2, s3, p))


def verify():
    grid = perimeter_one_grid()
    if len(grid) != 1176:
        raise AssertionError("The exact 0.01 grid must contain 1176 triangles.")

    maxima = {
        "l1_formula_vs_breakpoints": 0.0,
        "l1_formula_vs_lp": 0.0,
        "l2_formula_vs_continuous": 0.0,
        "linf_formula_vs_lp": 0.0,
        "two_branch_rule": 0.0,
        "l1_enumeration_two_branch_rule": 0.0,
        "feasibility_violation": 0.0,
        "reconstruction_error": 0.0,
    }
    total_branch_checks = 0
    total_global_checks = 0
    total_ties = 0
    setting_results = []

    for name, k1, k2 in settings():
        local = {key: 0.0 for key in maxima}
        ties = 0
        for triple in grid:
            per_norm_values = {1: [], 2: [], math.inf: []}
            l1_enumeration_values = []
            for branch in branches(triple):
                s1, s2, s3 = branch
                l1 = l1_formula(s1, s2, s3, k1, k2)
                l1_enum = l1_breakpoint_enumeration(s1, s2, s3, k1, k2)
                l1_enumeration_values.append(l1_enum[0])
                l1_reference = l1_lp(s1, s2, s3, k1, k2)
                l2 = l2_formula(s1, s2, s3, k1, k2)
                l2_reference = l2_continuous_reference(s1, s2, s3, k1, k2)
                linf = linf_formula(s1, s2, s3, k1, k2)
                linf_reference = linf_lp(s1, s2, s3, k1, k2)
                comparisons = (
                    ("l1_formula_vs_breakpoints", l1, l1_enum),
                    ("l1_formula_vs_lp", l1, l1_reference),
                    ("l2_formula_vs_continuous", l2, l2_reference),
                    ("linf_formula_vs_lp", linf, linf_reference),
                )
                for key, formula, reference in comparisons:
                    error = abs(formula[0] - reference[0])
                    local[key] = max(local[key], error)
                    maxima[key] = max(maxima[key], error)
                    if error > TOL:
                        raise AssertionError("{} mismatch: {}".format(
                            key, (triple, branch, formula[0], reference[0])))
                for p, answer in ((1, l1), (2, l2), (math.inf, linf)):
                    per_norm_values[p].append(answer[0])
                    violation = feasibility_violation(answer[1], answer[2], k1, k2)
                    residual_error = reconstruction_error(
                        answer[0], answer[1], answer[2], s1, s2, s3, p)
                    local["feasibility_violation"] = max(
                        local["feasibility_violation"], violation)
                    local["reconstruction_error"] = max(
                        local["reconstruction_error"], residual_error)
                    maxima["feasibility_violation"] = max(
                        maxima["feasibility_violation"], violation)
                    maxima["reconstruction_error"] = max(
                        maxima["reconstruction_error"], residual_error)
                total_branch_checks += 3
            kept = two_kept_branch_indices(triple)
            for p, values in per_norm_values.items():
                error = abs(min(values) - min(values[kept[0]], values[kept[1]]))
                local["two_branch_rule"] = max(local["two_branch_rule"], error)
                maxima["two_branch_rule"] = max(maxima["two_branch_rule"], error)
                if error > TOL:
                    raise AssertionError("Two-branch rule mismatch: {}".format(
                        (triple, p, values, kept)))
                total_global_checks += 1
            enumeration_error = abs(
                min(l1_enumeration_values)
                - min(l1_enumeration_values[kept[0]],
                      l1_enumeration_values[kept[1]]))
            local["l1_enumeration_two_branch_rule"] = max(
                local["l1_enumeration_two_branch_rule"], enumeration_error)
            maxima["l1_enumeration_two_branch_rule"] = max(
                maxima["l1_enumeration_two_branch_rule"], enumeration_error)
            if enumeration_error > TOL:
                raise AssertionError(
                    "Two-branch rule mismatch for L1 enumeration: {}".format(
                        (triple, l1_enumeration_values, kept)))
            total_global_checks += 1
            if sum(abs(value - min(per_norm_values[1])) <= TIE_TOL
                   for value in per_norm_values[1]) >= 2:
                ties += 1
                total_ties += 1
        setting_results.append({
            "setting": name, "k1": k1, "k2": k2, "grid_triangles": len(grid),
            "l1_tie_grid_points": ties, "max_errors": local,
        })

    return {
        "grid_increment": GRID_STEP,
        "perimeter": 1,
        "ordered_position_preserving_input": True,
        "strict_triangle_inequalities": True,
        "grid_triangles_per_setting": len(grid),
        "settings": setting_results,
        "total_branch_checks_per_norm": total_branch_checks,
        "total_global_checks": total_global_checks,
        "total_l1_ties": total_ties,
        "max_errors": maxima,
        "tie_tolerance": TIE_TOL,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "status": "all checks passed",
    }


def format_number(value):
    return "{:.3e}".format(value)


def write_tex(results):
    errors = results["max_errors"]
    labels = (
        r"$[1^\circ,90^\circ]$",
        r"$[45^\circ,90^\circ]$",
        r"$[30^\circ,50^\circ]$",
        r"$[90^\circ,120^\circ]$",
        r"$(k_1,k_2)=(3/4,5/4)$",
    )
    rows = []
    for label, setting in zip(labels, results["settings"]):
        setting_errors = setting["max_errors"]
        rows.append(
            "{} & {} & {} & {} & {} & {} \\\\".format(
                label,
                format_number(setting_errors["l1_formula_vs_breakpoints"]),
                format_number(setting_errors["l1_formula_vs_lp"]),
                format_number(setting_errors["l2_formula_vs_continuous"]),
                format_number(setting_errors["linf_formula_vs_lp"]),
                setting["l1_tie_grid_points"],
            )
        )
    text = r"""\newcommand{\VerificationGridTriangles}{%d}
\newcommand{\VerificationBranchChecks}{%d}
\newcommand{\VerificationGlobalChecks}{%d}
\newcommand{\VerificationLoneTies}{%d}
\newcommand{\VerificationLoneEnumerationError}{%s}
\newcommand{\VerificationLoneLpError}{%s}
\newcommand{\VerificationLtwoError}{%s}
\newcommand{\VerificationLinfError}{%s}
\newcommand{\VerificationTwoBranchError}{%s}
\newcommand{\VerificationLoneEnumerationTwoBranchError}{%s}
\newcommand{\VerificationFeasibilityViolation}{%s}
\newcommand{\VerificationReconstructionError}{%s}
\newcommand{\VerificationSettingRows}{%%
%s
}
""" % (
        results["grid_triangles_per_setting"],
        results["total_branch_checks_per_norm"],
        results["total_global_checks"],
        results["total_l1_ties"],
        format_number(errors["l1_formula_vs_breakpoints"]),
        format_number(errors["l1_formula_vs_lp"]),
        format_number(errors["l2_formula_vs_continuous"]),
        format_number(errors["linf_formula_vs_lp"]),
        format_number(errors["two_branch_rule"]),
        format_number(errors["l1_enumeration_two_branch_rule"]),
        format_number(errors["feasibility_violation"]),
        format_number(errors["reconstruction_error"]),
        "\n".join(rows),
    )
    (ROOT / "verification_summary.tex").write_text(text, encoding="utf-8")


def ternary_coordinates(data):
    """Map perimeter-one triples to an equilateral-triangle plotting frame."""
    data = np.asarray(data, dtype=float)
    return data[:, 1] + 0.5 * data[:, 2], SQRT3_2 * data[:, 2]


def draw_ternary_frame(ax):
    """Draw an English-labelled perimeter-one ternary frame."""
    for tick in np.arange(0.1, 1.0, 0.1):
        ax.plot([1 - tick, 0.5 * (1 - tick)],
                [0, SQRT3_2 * (1 - tick)], color="0.88", lw=0.35, zorder=1)
        ax.plot([tick, tick + 0.5 * (1 - tick)],
                [0, SQRT3_2 * (1 - tick)], color="0.88", lw=0.35, zorder=1)
        ax.plot([0.5 * tick, 1 - 0.5 * tick], [SQRT3_2 * tick] * 2,
                color="0.88", lw=0.35, zorder=1)
    ax.plot([0, 1, 0.5, 0], [0, 0, SQRT3_2, 0], color="black", lw=0.9)
    valid = np.array(((0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)))
    x, y = ternary_coordinates(valid)
    ax.plot(np.r_[x, x[0]], np.r_[y, y[0]], "--", color="0.35", lw=0.8,
            zorder=10)
    ax.text(0.5, -0.06, "Input side b", ha="center", va="top", fontsize=11)
    ax.text(0.12, 0.43, "Input side a", ha="center", fontsize=11)
    ax.text(0.88, 0.43, "Input side c", ha="center", fontsize=11)
    ax.text(0.5, 0.82, "Input perimeter = 1", ha="center", fontsize=10)
    ax.set(xlim=(-0.06, 1.06), ylim=(-0.09, 0.91), aspect="equal")
    ax.axis("off")


def triangular_mesh(subdivisions=140, inset=1.0e-4):
    """Return a dense mesh strictly inside the nondegenerate input domain."""
    points = []
    for i in range(subdivisions + 1):
        for j in range(subdivisions + 1 - i):
            a = 0.5 * (1 - j / subdivisions)
            b = 0.5 * ((i + j) / subdivisions)
            c = 0.5 * (1 - i / subdivisions)
            points.append((a, b, c))
    points = np.asarray(points, dtype=float)
    return (1.0 - inset) * points + inset / 3.0


def global_candidate_solution(triple, k1, k2, p):
    """Evaluate the three sorted branches and retain their optimality mask."""
    a, b, c = sorted(triple)
    values = [branch_formula(*branch, k1, k2, p)[0]
              for branch in ((b, c, a), (a, c, b), (a, b, c))]
    best = min(values)
    tolerance = TIE_TOL * max(1.0, abs(best))
    indices = [index for index, value in enumerate(values)
               if abs(value - best) <= tolerance]
    label = ("I", "II", "III")[indices[0]] if len(indices) == 1 else "Tie"
    return best, label


def plot_multinorm_figures():
    """Generate English \ell_2 and \ell_infty decision and cost maps for [1,90]."""
    _, k1, k2 = settings()[0]
    exact_grid = np.asarray(perimeter_one_grid(), dtype=float)
    dense_mesh = triangular_mesh()
    for p, stem, norm_label in (
            (2, "l2", r"$\ell_2$"),
            (math.inf, "linf", r"$\ell_\infty$")):
        dense_values = np.asarray([
            global_candidate_solution(point, k1, k2, p)[0]
            for point in dense_mesh
        ])
        exact_solutions = [
            global_candidate_solution(point, k1, k2, p) for point in exact_grid
        ]
        exact_values = np.asarray([solution[0] for solution in exact_solutions])
        exact_labels = [solution[1] for solution in exact_solutions]

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1))
        x, y = ternary_coordinates(dense_mesh)
        dense_labels = [
            global_candidate_solution(point, k1, k2, p)[1] for point in dense_mesh
        ]
        for label, color in BRANCH_COLORS.items():
            selected = np.asarray([item == label for item in dense_labels])
            axes[0].scatter(x[selected], y[selected], s=2.4, color=color,
                            edgecolors="none", rasterized=True, label=label,
                            zorder=2)
        x, y = ternary_coordinates(exact_grid)
        for label, color in BRANCH_COLORS.items():
            selected = np.asarray([item == label for item in exact_labels])
            axes[1].scatter(x[selected], y[selected], s=6, color=color,
                            edgecolors="none", rasterized=True, label=label,
                            zorder=2)
        for ax, title in zip(
                axes,
                ("(a) Closed-form candidate rule: dense mesh",
                 "(b) Closed-form candidate rule: exact 0.01 input lattice")):
            draw_ternary_frame(ax)
            ax.set_title(title, fontsize=12, pad=8)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, ["Branch " + label if label != "Tie" else "Tied branches"
                             for label in labels], loc="lower center", ncol=4,
                   frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=11)
        fig.suptitle(
            r"Optimal branch distribution for {} under $[1^\circ,90^\circ]$"
            "; branches are defined after sorting the input sides".format(norm_label),
            fontsize=13, y=0.98)
        fig.tight_layout(rect=(0, 0.10, 1, 0.91))
        fig.savefig(ROOT / ("1-90_{}_decision_regions.png".format(stem)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1))
        vmin = min(float(dense_values.min()), float(exact_values.min()))
        vmax = max(float(dense_values.max()), float(exact_values.max()))
        if not vmax > vmin:
            vmax = np.nextafter(vmin, math.inf)
        norm = Normalize(vmin=vmin, vmax=vmax)
        x, y = ternary_coordinates(dense_mesh)
        contour = axes[0].tricontourf(
            Triangulation(x, y), dense_values, levels=25, cmap="viridis",
            norm=norm, zorder=2)
        x, y = ternary_coordinates(exact_grid)
        axes[1].scatter(x, y, c=exact_values, s=6, cmap="viridis", norm=norm,
                        edgecolors="none", rasterized=True, zorder=2)
        for ax, title in zip(
                axes,
                ("(a) Closed-form candidate rule: dense mesh",
                 "(b) Closed-form candidate rule: exact 0.01 input lattice")):
            draw_ternary_frame(ax)
            ax.set_title(title, fontsize=12, pad=8)
        colorbar = fig.colorbar(contour, ax=axes, fraction=0.035, pad=0.03)
        colorbar.set_label("Optimal {} cost".format(norm_label), fontsize=12)
        fig.suptitle(
            r"Optimal cost distribution for {} under $[1^\circ,90^\circ]$"
            "; the two panels share one color scale".format(norm_label),
            fontsize=13, y=0.98)
        fig.tight_layout(rect=(0, 0, 0.93, 0.91))
        fig.savefig(ROOT / ("1-90_{}_costs.png".format(stem)), dpi=300,
                    bbox_inches="tight")
        plt.close(fig)


def main():
    results = verify()
    (ROOT / "verification_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_tex(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
