#!/usr/bin/env python3
"""Render the README comparison figures from committed experiment outputs.

This script uses only the Python standard library. Run it from any directory:

    python scripts/render_readme_figures.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring


ROOT = Path(__file__).resolve().parents[1]
SVG = "http://www.w3.org/2000/svg"
COLORS = {
    "highest_allowed": "#2458a6",
    "robust_mpc": "#2e8b78",
    "log_dqn": "#e29a36",
    "vmaf_dqn": "#8957a5",
    "learned": "#8957a5",
    "harmonic": "#2e8b78",
}


def add(parent: Element, tag: str, **attrs: str | int | float) -> Element:
    return SubElement(parent, f"{{{SVG}}}{tag}", {key.replace("_", "-"): str(value) for key, value in attrs.items()})


def label(parent: Element, x: float, y: float, value: str, *, size: int = 14,
          fill: str = "#243247", anchor: str = "middle", weight: str = "normal") -> None:
    node = add(parent, "text", x=x, y=y, fill=fill, font_size=size,
               text_anchor=anchor, font_family="Arial, sans-serif", font_weight=weight)
    node.text = value


def svg_root(width: int, height: int, title: str, description: str) -> Element:
    root = Element(f"{{{SVG}}}svg", {
        "viewBox": f"0 0 {width} {height}",
        "width": str(width),
        "height": str(height),
        "role": "img",
    })
    title_node = SubElement(root, f"{{{SVG}}}title")
    title_node.text = title
    desc_node = SubElement(root, f"{{{SVG}}}desc")
    desc_node.text = description
    add(root, "rect", x=0, y=0, width=width, height=height, fill="#ffffff", rx=18)
    return root


def write_svg(path: Path, root: Element) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tostring(root, encoding="utf-8", xml_declaration=True))


def read_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def vmaf_groups() -> dict[str, dict[str, tuple[float, float, float]]]:
    result = read_json("runs/vmaf-reward-stress-transfer-v1/vmaf/vmaf_transfer.json")
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in result["summaries"]:
        name = row["policy"]
        if name == "v3_highest_allowed":
            group = "highest_allowed"
        elif name == "v3_robust_mpc":
            group = "robust_mpc"
        elif name.startswith("v3_log_reward_seed_"):
            group = "log_dqn"
        elif name.startswith("v3_vmaf_reward_seed_"):
            group = "vmaf_dqn"
        else:
            continue
        values[row["content"]][group].append(row["mean_case_vmaf"])
    return {
        content: {
            group: (sum(scores) / len(scores), min(scores), max(scores))
            for group, scores in groups.items()
        }
        for content, groups in values.items()
    }


def render_vmaf_comparison() -> None:
    content = vmaf_groups()
    width, height = 1120, 570
    root = svg_root(
        width, height,
        "VMAF comparison across two video contents",
        "Mean VMAF on frozen controlled-capacity test profiles. DQN values average three seeds; whiskers show the minimum and maximum seed means.",
    )
    label(root, width / 2, 40, "Playback quality: learned policies did not beat the references consistently", size=22, weight="bold")
    label(root, width / 2, 67, "Mean VMAF across 12 frozen stress profiles per content; DQN markers show the three-seed range", size=13, fill="#58677d")

    groups = ["highest_allowed", "robust_mpc", "log_dqn", "vmaf_dqn"]
    names = ["Highest allowed", "RobustMPC", "Log-reward DQN", "VMAF-reward DQN"]
    palette = [COLORS[group] for group in groups]
    panels = [("envivio", "Envivio", 55), ("sintel", "Sintel", 585)]
    plot_top, plot_bottom, plot_height = 120, 430, 310
    maximum = 70.0

    for key, title, left in panels:
        plot_left, plot_width = left + 45, 430
        label(root, left + 260, 103, title, size=18, weight="bold")
        for tick in range(0, 71, 10):
            y = plot_bottom - tick / maximum * plot_height
            add(root, "line", x1=plot_left, y1=y, x2=plot_left + plot_width, y2=y,
                stroke="#dce3ec", stroke_width=1)
            label(root, plot_left - 10, y + 5, str(tick), size=11, anchor="end", fill="#64748b")
        add(root, "line", x1=plot_left, y1=plot_top, x2=plot_left, y2=plot_bottom,
            stroke="#718096", stroke_width=1.2)
        add(root, "line", x1=plot_left, y1=plot_bottom, x2=plot_left + plot_width,
            y2=plot_bottom, stroke="#718096", stroke_width=1.2)
        label(root, left + 8, (plot_top + plot_bottom) / 2, "VMAF", size=12, fill="#58677d")
        centers = [plot_left + plot_width * (i + 0.5) / 4 for i in range(4)]
        for idx, (group, name, color) in enumerate(zip(groups, names, palette)):
            mean, low, high = content[key][group]
            x = centers[idx] - 28
            y = plot_bottom - mean / maximum * plot_height
            add(root, "rect", x=x, y=y, width=56, height=plot_bottom - y, rx=5, fill=color)
            label(root, centers[idx], y - 9, f"{mean:.1f}", size=12, weight="bold")
            if group in ("log_dqn", "vmaf_dqn"):
                y_low = plot_bottom - low / maximum * plot_height
                y_high = plot_bottom - high / maximum * plot_height
                add(root, "line", x1=centers[idx], y1=y_high, x2=centers[idx], y2=y_low,
                    stroke="#243247", stroke_width=2)
                add(root, "line", x1=centers[idx] - 7, y1=y_high, x2=centers[idx] + 7, y2=y_high,
                    stroke="#243247", stroke_width=2)
                add(root, "line", x1=centers[idx] - 7, y1=y_low, x2=centers[idx] + 7, y2=y_low,
                    stroke="#243247", stroke_width=2)
            label(root, centers[idx], plot_bottom + 22, name, size=11)
    label(root, width / 2, 505,
          "Three-seed spread is visible: reward changes helped some seeds and hurt others. These are simulator results, not viewer studies.",
          size=13, fill="#58677d")
    write_svg(ROOT / "docs/images/vmaf-policy-comparison.svg", root)


def render_bola_comparison() -> None:
    result = read_json("runs/single-task-vod-v1/vmaf_transfer.json")
    values = {
        (row["content"], row["policy"]): row["mean_case_vmaf"]
        for row in result["summaries"]
        if row["policy"] in ("bola_basic", "bola_basic_filtered", "highest_allowed", "old_frozen")
    }
    policy_names = ["BOLA-Basic", "BOLA + safety", "Highest allowed", "Frozen DQN"]
    policy_ids = ["bola_basic", "bola_basic_filtered", "highest_allowed", "old_frozen"]
    colors = ["#2e8b78", "#83b7aa", COLORS["highest_allowed"], COLORS["log_dqn"]]
    width, height = 1120, 570
    root = svg_root(
        width, height,
        "BOLA and DQN VMAF comparison",
        "Mean VMAF across four historical ferry sessions per video. BOLA Basic beat highest allowed on Envivio and scored lower on Sintel. The frozen DQN exactly matched highest allowed.",
    )
    label(root, width / 2, 40, "BOLA’s result changed with the video content", size=22, weight="bold")
    label(root, width / 2, 67, "Mean VMAF across four historical ferry sessions per content; separate from the controlled-capacity study", size=13, fill="#58677d")
    panels = [("envivio", "Envivio", 55), ("sintel", "Sintel", 585)]
    plot_top, plot_bottom, plot_height, maximum = 120, 430, 310, 85.0
    for content_key, title, left in panels:
        plot_left, plot_width = left + 45, 430
        label(root, left + 260, 103, title, size=18, weight="bold")
        for tick in range(0, 81, 10):
            y = plot_bottom - tick / maximum * plot_height
            add(root, "line", x1=plot_left, y1=y, x2=plot_left + plot_width, y2=y,
                stroke="#dce3ec", stroke_width=1)
            label(root, plot_left - 10, y + 5, str(tick), size=11, anchor="end", fill="#64748b")
        add(root, "line", x1=plot_left, y1=plot_top, x2=plot_left, y2=plot_bottom,
            stroke="#718096", stroke_width=1.2)
        add(root, "line", x1=plot_left, y1=plot_bottom, x2=plot_left + plot_width,
            y2=plot_bottom, stroke="#718096", stroke_width=1.2)
        centers = [plot_left + plot_width * (i + 0.5) / 4 for i in range(4)]
        for idx, (policy_id, name, color) in enumerate(zip(policy_ids, policy_names, colors)):
            value = values[(content_key, policy_id)]
            x, center = centers[idx] - 28, centers[idx]
            y = plot_bottom - value / maximum * plot_height
            add(root, "rect", x=x, y=y, width=56, height=plot_bottom - y, rx=5, fill=color)
            label(root, center, y - 9, f"{value:.1f}", size=12, weight="bold")
            label(root, center, plot_bottom + 22, name, size=11)
    label(root, width / 2, 505,
          "This small, same-route screen is a mixed result; it does not rank BOLA universally or reproduce a production DASH controller.",
          size=13, fill="#58677d")
    write_svg(ROOT / "docs/images/bola-vmaf-comparison.svg", root)


def render_predictor_comparison() -> None:
    result = read_json("runs/throughput-predictor-v1/ghent-measured-results.json")
    forecast = result["forecast_accuracy"]
    policy = {row["policy"]: row["summary"] for row in result["policies"]}
    values = [
        ("Highest allowed", policy["highest_allowed"]["decision_latency_p50_ms"], COLORS["highest_allowed"]),
        ("Harmonic RobustMPC", policy["harmonic_mean_robust_mpc"]["decision_latency_p50_ms"], COLORS["harmonic"]),
        ("Learned + RobustMPC", policy["learned_forecast_robust_mpc"]["decision_latency_p50_ms"], COLORS["learned"]),
    ]
    width, height = 1120, 470
    root = svg_root(
        width, height,
        "Throughput predictor comparison on historical Ghent traces",
        "On 1,568 forecast decisions, the learned predictor reduced mean absolute error from 13.88 to 10.54 Mbps. Playback completed without stalls and tied the highest-allowed rule on 31 of 32 sessions, while RobustMPC decision latency was about 129 milliseconds.",
    )
    label(root, width / 2, 40, "Better forecast error did not improve playback outcomes", size=22, weight="bold")
    label(root, width / 2, 67, "32 historical Ghent sessions; 80 ms RTT was assumed", size=13, fill="#58677d")

    # Panel one: forecast MAE, with bars on a common zero-based scale.
    left, top, chart_w, chart_h = 95, 125, 390, 180
    label(root, left + chart_w / 2, 105, "Forecast error (lower is better)", size=16, weight="bold")
    max_mae = 16.0
    for tick in range(0, 17, 4):
        y = top + chart_h - tick / max_mae * chart_h
        add(root, "line", x1=left, y1=y, x2=left + chart_w, y2=y, stroke="#dce3ec", stroke_width=1)
        label(root, left - 12, y + 5, str(tick), size=11, anchor="end", fill="#64748b")
    forecast_values = [
        ("Harmonic", forecast["harmonic_robust_point"]["mae_mbps"], COLORS["harmonic"]),
        ("Learned p20", forecast["learned_p20"]["mae_mbps"], COLORS["learned"]),
    ]
    for i, (name, val, color) in enumerate(forecast_values):
        x = left + 72 + i * 165
        bar_h = val / max_mae * chart_h
        y = top + chart_h - bar_h
        add(root, "rect", x=x, y=y, width=72, height=bar_h, rx=5, fill=color)
        label(root, x + 36, y - 9, f"{val:.2f} Mbps", size=12, weight="bold")
        label(root, x + 36, top + chart_h + 22, name, size=12)

    # Panel two: log-scaled decision latency so the simple rule remains visible.
    right, latency_top, latency_w, latency_h = 605, 133, 445, 185
    label(root, right + latency_w / 2, 105, "Median decision latency (log scale)", size=16, weight="bold")
    min_log, max_log = -2.5, 2.3
    for tick_value in (0.01, 0.1, 1, 10, 100):
        import math
        position = (math.log10(tick_value) - min_log) / (max_log - min_log)
        x = right + position * latency_w
        add(root, "line", x1=x, y1=latency_top, x2=x, y2=latency_top + latency_h,
            stroke="#dce3ec", stroke_width=1)
        label(root, x, latency_top + latency_h + 18, f"{tick_value:g}", size=10, fill="#64748b")
    for i, (name, val, color) in enumerate(values):
        y = latency_top + 25 + i * 51
        position = (math.log10(val) - min_log) / (max_log - min_log)
        bar_width = position * latency_w
        add(root, "rect", x=right, y=y, width=bar_width, height=26, rx=5, fill=color)
        label(root, right + 6, y - 5, name, size=11, anchor="start", weight="bold")
        label(root, right + min(bar_width + 8, latency_w - 5), y + 18, f"{val:.3g} ms",
              size=11, anchor="start", fill="#243247")
    label(root, right + latency_w, latency_top + latency_h + 38, "ms", size=10, anchor="end", fill="#64748b")

    add(root, "rect", x=95, y=390, width=955, height=48, rx=10, fill="#f2f6fb")
    label(root, width / 2, 420,
          "All policies completed 32/32 sessions with zero stalls. Learned RobustMPC tied the simple rule on 31/32; one session was worse.",
          size=13, fill="#34445a")
    write_svg(ROOT / "docs/images/predictor-vs-playback.svg", root)


if __name__ == "__main__":
    render_vmaf_comparison()
    render_bola_comparison()
    render_predictor_comparison()
