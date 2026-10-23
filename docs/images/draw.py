"""Draws the README's diagrams, each in a light and a dark version for GitHub's two themes, and the
banner, which carries its own background:

    uv run python docs/images/draw.py
"""

import math
from pathlib import Path

HERE = Path(__file__).parent
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"

# GitHub's own greys, and the menus' accent (darker on white, so text in it stays readable).
THEMES = {
    "light": {
        "text": "#1f2328",
        "muted": "#59636e",
        "card": "#f6f8fa",
        "line": "#d1d9e0",
        "chip": "#eaeef2",
        "accent": "#0a7d5f",
        "accent-soft": "#d7f4e9",
        "warn": "#bc4c00",
        "warn-soft": "#fff1e5",
    },
    "dark": {
        "text": "#e6edf3",
        "muted": "#9198a1",
        "card": "#151b23",
        "line": "#3d444d",
        "chip": "#262c36",
        "accent": "#5fd7af",
        "accent-soft": "#12352b",
        "warn": "#f0883e",
        "warn-soft": "#3d2513",
    },
}


def svg(width: int, height: int, title: str, body: str, colours: dict[str, str] | None = None) -> str:
    classes = ""
    if colours:
        fills = "\n".join(f"  .{name} {{ fill: {value}; }}" for name, value in colours.items())
        strokes = "\n".join(f"  .s-{name} {{ stroke: {value}; }}" for name, value in colours.items())
        classes = f"{fills}\n{strokes}\n"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" \
height="{height}" role="img" aria-label="{title}">
<title>{title}</title>
<style>
  text {{ font-family: {SANS}; }}
  .mono {{ font-family: {MONO}; }}
{classes}</style>
{body}
</svg>
"""


def label(
    x: float, y: float, words: str, cls: str, size: float, anchor: str = "start", weight: int = 400
) -> str:
    return (
        f'<text x="{x}" y="{y}" class="{cls}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}">{words}</text>'
    )


def arrowhead(cls: str) -> str:
    return (
        f'<marker id="head-{cls}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="{cls}"/></marker>'
    )


# Why: what candor scores is the file you run.


def flow(x: int, title: str, steps: list[str], note: str, good: bool) -> str:
    width, ys = 488, [98, 150, 202]
    tone = "accent" if good else "warn"
    parts = [
        f'<rect x="{x + 0.75}" y="0.75" width="{width - 1.5}" height="258.5" rx="14" '
        f'class="card s-{"accent" if good else "line"}" stroke-width="1.5"/>',
        label(x + 28, 50, title, "text", 19, weight=700),
        f'<line x1="{x + 42}" y1="{ys[0]}" x2="{x + 42}" y2="{ys[-1]}" class="s-line" stroke-width="2"/>',
    ]
    for i, (y, words) in enumerate(zip(ys, steps, strict=True)):
        last = i == len(ys) - 1
        if not last:
            parts += [
                f'<circle cx="{x + 42}" cy="{y}" r="13" class="{"accent-soft" if good else "chip"}"/>',
                label(x + 42, y + 4.5, str(i + 1), "accent" if good else "muted", 13, "middle", 700),
                label(x + 70, y + 5.5, words, "text", 15.5),
            ]
            continue
        mark = (
            f"M{x + 36},{y} l4.5,4.5 l8,-9"
            if good
            else f"M{x + 37.5},{y - 4.5} l9,9 M{x + 46.5},{y - 4.5} l-9,9"
        )
        parts += [
            f'<circle cx="{x + 42}" cy="{y}" r="13" class="{tone}"/>',
            f'<path d="{mark}" fill="none" class="s-card" stroke-width="2.6" stroke-linecap="round" '
            'stroke-linejoin="round"/>',
            label(x + 70, y + 5.5, words, tone, 15.5, weight=700),
        ]
    parts.append(label(x + 28, 240, note, "muted", 13.5))
    return "\n".join(parts)


def why(colours: dict[str, str]) -> str:
    body = "\n".join(
        [
            flow(
                0,
                "Scored in PyTorch",
                [
                    "Edits scored on the 16-bit model",
                    "The best quantised to GGUF",
                    "The file you run: not the one scored",
                ],
                "Quantising can round a small edit away.",
                good=False,
            ),
            flow(
                512,
                "Scored by candor",
                [
                    "Edits scored in llama-server, on your GGUF",
                    "The best written in, checked weight by weight",
                    "The file you run: the one measured",
                ],
                "At your settings: sampling, thinking budget, KV cache.",
                good=True,
            ),
        ]
    )
    return svg(1000, 260, "What candor scores is the file you run", body, colours)


# How it works: the four stages.

STAGES = [
    ("Read directions", ["harmful vs harmless prompts,", "mean difference per layer"], "candor run"),
    ("Build adapters", ["rank-1 LoRA adapters,", "one per edited weight"], "candor run"),
    ("Search", ["every adapter in one server;", "a trial only sets strengths"], "candor search"),
    ("Export and measure", ["written into your GGUF,", "then that exact file measured"], "candor export"),
]


def icon(stage: int, cx: float, cy: float) -> str:
    if stage == 0:
        harmful = [(-38, -10), (-26, 2), (-39, 12)]
        harmless = [(28, -11), (40, 0), (30, 11)]
        dots = [f'<circle cx="{cx + dx}" cy="{cy + dy}" r="5" class="warn"/>' for dx, dy in harmful]
        dots += [f'<circle cx="{cx + dx}" cy="{cy + dy}" r="5" class="muted"/>' for dx, dy in harmless]
        return "\n".join(
            [
                *dots,
                f'<line x1="{cx + 17}" y1="{cy}" x2="{cx - 15}" y2="{cy}" class="s-accent" '
                'stroke-width="2.5" stroke-linecap="round" marker-end="url(#head-accent)"/>',
            ]
        )
    if stage == 1:
        rows = []
        for i in range(3):
            y = cy - 21 + i * 15
            rows += [
                f'<rect x="{cx - 36}" y="{y}" width="54" height="11" rx="3" class="chip s-line"/>',
                f'<line x1="{cx + 18}" y1="{y + 5.5}" x2="{cx + 25}" y2="{y + 5.5}" class="s-accent" '
                'stroke-width="2"/>',
                f'<rect x="{cx + 25}" y="{y}" width="11" height="11" rx="2.5" class="accent"/>',
            ]
        return "\n".join(rows)
    if stage == 2:
        # Refusals up, KL across: the pick is the least KL under the cap.
        dots = [(-26, -14), (-12, -6), (0, -18), (12, -4), (26, -12), (4, 12), (18, 14), (28, 8)]
        return "\n".join(
            [
                f'<path d="M{cx - 38},{cy - 26} V{cy + 22} H{cx + 38}" fill="none" class="s-muted" '
                'stroke-width="2" stroke-linecap="round"/>',
                f'<line x1="{cx - 34}" y1="{cy + 2}" x2="{cx + 36}" y2="{cy + 2}" class="s-warn" '
                'stroke-width="1.5" stroke-dasharray="4 3"/>',
                *(f'<circle cx="{cx + dx}" cy="{cy + dy}" r="3.5" class="muted"/>' for dx, dy in dots),
                f'<circle cx="{cx - 14}" cy="{cy + 11}" r="7.5" fill="none" class="s-accent" '
                'stroke-width="2"/>',
                f'<circle cx="{cx - 14}" cy="{cy + 11}" r="3.5" class="accent"/>',
            ]
        )
    return "\n".join(
        [
            f'<path d="M{cx - 20},{cy - 26} H{cx + 8} L{cx + 20},{cy - 14} V{cy + 26} H{cx - 20} Z" '
            'class="card s-text" stroke-width="2" stroke-linejoin="round"/>',
            f'<path d="M{cx + 8},{cy - 26} V{cy - 14} H{cx + 20}" fill="none" class="s-text" '
            'stroke-width="2" stroke-linejoin="round"/>',
            f'<path d="M{cx - 10},{cy + 3} l7,7 l13,-15" fill="none" class="s-accent" stroke-width="3" '
            'stroke-linecap="round" stroke-linejoin="round"/>',
        ]
    )


def pipeline(colours: dict[str, str]) -> str:
    width, gap, height = 229, 28, 226
    parts = [f"<defs>{arrowhead('accent')}{arrowhead('muted')}</defs>"]
    for i, (title, lines, command) in enumerate(STAGES):
        x = i * (width + gap)
        cx = x + width / 2
        chip = len(command) * 7.6 + 24
        parts += [
            f'<rect x="{x + 0.75}" y="0.75" width="{width - 1.5}" height="{height - 1.5}" rx="12" '
            'class="card s-line" stroke-width="1.5"/>',
            f'<circle cx="{x + 26}" cy="26" r="13" class="accent-soft"/>',
            label(x + 26, 30.5, str(i + 1), "accent", 13, "middle", 700),
            icon(i, cx, 68),
            label(cx, 124, title, "text", 16.5, "middle", 700),
            label(cx, 148, lines[0], "muted", 13.5, "middle"),
            label(cx, 167, lines[1], "muted", 13.5, "middle"),
            f'<rect x="{cx - chip / 2}" y="184" width="{chip}" height="26" rx="6" class="chip"/>',
            f'<text x="{cx}" y="201.5" class="mono text" font-size="12.5" '
            f'text-anchor="middle">{command}</text>',
        ]
        if i < len(STAGES) - 1:
            parts.append(
                f'<line x1="{x + width + 5}" y1="113" x2="{x + width + gap - 6}" y2="113" class="s-muted" '
                'stroke-width="2" marker-end="url(#head-muted)"/>'
            )
    return svg(4 * width + 3 * gap, height, "How candor works, in four stages", "\n".join(parts), colours)


# The banner: an activation losing its part along the refusal direction.


def banner() -> str:
    pills = ["llama.cpp", "quantised GGUF", "thinking models", "answers never printed"]
    x, chips = 60, []
    for words in pills:
        width = len(words) * 7.5 + 28
        chips += [
            f'<rect x="{x}" y="204" width="{width}" height="30" rx="15" fill="#5fd7af" fill-opacity="0.08" '
            'stroke="#5fd7af" stroke-opacity="0.45"/>',
            f'<text x="{x + width / 2}" y="224" fill="#a7ecd5" font-size="13.5" '
            f'text-anchor="middle">{words}</text>',
        ]
        x += width + 10
    ox, oy, tx, ty = 742, 200, 900, 70
    angle = math.degrees(math.atan2(ty - oy, tx - ox))
    body = f"""<defs>
  <linearGradient id="sky" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#0d1117"/><stop offset="0.55" stop-color="#0f1c22"/>
    <stop offset="1" stop-color="#0c3128"/>
  </linearGradient>
  <radialGradient id="glow" cx="0.82" cy="0.5" r="0.45">
    <stop offset="0" stop-color="#5fd7af" stop-opacity="0.16"/>
    <stop offset="1" stop-color="#5fd7af" stop-opacity="0"/>
  </radialGradient>
  <pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse">
    <circle cx="2" cy="2" r="1" fill="#ffffff" fill-opacity="0.06"/>
  </pattern>
  <marker id="tip" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fd7af"/>
  </marker>
  <marker id="tip-grey" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
    <path d="M0,0 L10,5 L0,10 z" fill="#8b949e"/>
  </marker>
  <clipPath id="frame"><rect width="1000" height="260" rx="18"/></clipPath>
</defs>
<g clip-path="url(#frame)">
  <rect width="1000" height="260" fill="url(#sky)"/>
  <rect width="1000" height="260" fill="url(#dots)"/>
  <rect width="1000" height="260" fill="url(#glow)"/>
</g>
<text x="56" y="108" fill="#f0f6fc" font-size="76" font-weight="800" letter-spacing="-2">candor</text>
<text x="60" y="148" fill="#c9d1d9" font-size="22">Remove a model's refusal direction,</text>
<text x="60" y="178" fill="#5fd7af" font-size="22" font-weight="600">\
scored the way you actually run it.</text>
{chr(10).join(chips)}
<line x1="{ox}" y1="{oy + 14}" x2="{ox}" y2="38" stroke="#f0883e" stroke-opacity="0.55" stroke-width="1.5" \
stroke-dasharray="5 5"/>
<text x="{ox - 10}" y="34" fill="#f0883e" font-size="13" text-anchor="end">refusal direction</text>
<line x1="{tx}" y1="{ty}" x2="{tx}" y2="{oy}" stroke="#f0883e" stroke-width="2" stroke-dasharray="5 4"/>
<text x="{tx + 12}" y="{(ty + oy) / 2 + 4}" fill="#f0883e" font-size="13">removed</text>
<path d="M{tx - 11},{oy} V{oy - 11} H{tx}" fill="none" stroke="#8b949e" stroke-width="1.2"/>
<line x1="{ox}" y1="{oy}" x2="{tx - 3}" y2="{ty + 3}" stroke="#8b949e" stroke-width="2.5" \
marker-end="url(#tip-grey)"/>
<text transform="translate({(ox + tx) / 2 - 7} {(oy + ty) / 2 - 9}) rotate({angle:.1f})" fill="#8b949e" \
font-size="13" text-anchor="middle">activation</text>
<line x1="{ox}" y1="{oy}" x2="{tx - 4}" y2="{oy}" stroke="#5fd7af" stroke-width="3.5" marker-end="url(#tip)"/>
<text x="{(ox + tx) / 2}" y="{oy + 26}" fill="#5fd7af" font-size="13" text-anchor="middle">edited</text>
<circle cx="{ox}" cy="{oy}" r="4" fill="#f0f6fc"/>"""
    title = "candor: remove a model's refusal direction, scored the way you actually run it"
    return svg(1000, 260, title, body)


def main() -> None:
    (HERE / "banner.svg").write_text(banner())
    for theme, colours in THEMES.items():
        (HERE / f"why-{theme}.svg").write_text(why(colours))
        (HERE / f"pipeline-{theme}.svg").write_text(pipeline(colours))


if __name__ == "__main__":
    main()
