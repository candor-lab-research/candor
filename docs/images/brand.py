"""Draws candor's brand files (docs/brand.md says how to use them), every letter as an outline so
they look the same everywhere, and renders the PNGs GitHub and Hugging Face need with Chrome
(`CANDOR_CHROME` points at another build):

    uv run --with fonttools --with uharfbuzz python docs/images/brand.py

The mark is a disc of stripes, a representation split into its directions, with the middle one
pulled out along itself: the one direction candor removes. The name is set in Source Serif 4 and
the rest in Inter, both under the SIL Open Font License, fetched once from Google Fonts' repository
at a pinned commit into candor's cache.
"""

import hashlib
import math
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

HERE = Path(__file__).parent
CACHE = Path(os.environ.get("CANDOR_CACHE", Path.home() / ".cache" / "candor")) / "fonts"
CHROME = os.environ.get("CANDOR_CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
FONTS_AT = "https://raw.githubusercontent.com/google/fonts/e7db6ca603fd4862010383ff17ffd6cc4ffac1c0/ofl"
FONTS = {
    "SourceSerif4[opsz,wght].ttf": (
        "sourceserif4/SourceSerif4%5Bopsz,wght%5D.ttf",
        "97b2d4da6e3cb494b5a1e66ae176914d852ccabef49e0c02c0df25f3e39aca0b",
    ),
    "Inter[opsz,wght].ttf": (
        "inter/Inter%5Bopsz,wght%5D.ttf",
        "29160a80ff49ddcab2c97711247e08b1fab27a484a329ce8b813d820dc559031",
    ),
}
NAME = ("SourceSerif4[opsz,wght].ttf", {"opsz": 60, "wght": 560}, -0.01)
TEXT = ("Inter[opsz,wght].ttf", {"opsz": 20, "wght": 400})
CAPS = ("Inter[opsz,wght].ttf", {"opsz": 20, "wght": 500})

INK, WHITE, MINT, GREY = "#0d1117", "#f0f6fc", "#5fd7af", "#c9d1d9"
# The stripes' and name's colour, and the accent's, on each kind of background.
ON = {"dark": (WHITE, MINT), "light": (INK, "#0a7d5f")}
STRIPES, GAP = 7, 0.55  # a gap is this many stripes wide
PULL = 0.2  # how far the middle stripe is pulled out, in disc widths


def font(name: str) -> Path:
    path = CACHE / name
    where, sha256 = FONTS[name]
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{FONTS_AT}/{where}") as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise SystemExit(f"{name} from Google Fonts doesn't match its pinned sha256")
        path.write_bytes(data)
    return path


def typeset(file: str, axes: dict[str, float], text: str, size: float, tracking: float = 0.0):
    """The text as one outline at a size in px, its baseline at y=0, with its width, x-height and
    top (above the baseline, so negative)."""
    path = font(file)
    static = instantiateVariableFont(TTFont(path), axes)
    shaper = hb.Font(hb.Face(hb.Blob.from_file_path(str(path))))
    shaper.set_variations(axes)
    buffer = hb.Buffer()
    buffer.add_str(text)
    buffer.guess_segment_properties()
    hb.shape(shaper, buffer, {"kern": True})
    glyphs, order = static.getGlyphSet(), static.getGlyphOrder()
    em = static["head"].unitsPerEm
    scale = size / em
    pen = SVGPathPen(glyphs, ntos=lambda v: f"{v:.2f}".rstrip("0").rstrip("."))
    bounds = BoundsPen(glyphs)
    x = 0.0
    for info, position in zip(buffer.glyph_infos, buffer.glyph_positions, strict=True):
        at = (scale, 0, 0, -scale, (x + position.x_offset) * scale, -position.y_offset * scale)
        glyphs[order[info.codepoint]].draw(TransformPen(pen, at))
        glyphs[order[info.codepoint]].draw(TransformPen(bounds, at))
        x += position.x_advance + tracking * em
    width, x_height = (x - tracking * em) * scale, static["OS/2"].sxHeight * scale
    return pen.getCommands(), width, x_height, bounds.bounds[1]


def mark(cx: float, cy: float, r: float, on: str = "dark") -> str:
    """The disc's stripes, centred on cx,cy, with the middle stripe pulled out to the right."""
    stripe = 2 * r / (STRIPES + (STRIPES - 1) * GAP)
    fill, accent = ON[on]
    parts = []
    for i in range(STRIPES):
        top = -r + i * stripe * (1 + GAP)
        bottom = top + stripe
        a, b = (math.sqrt(max(r * r - y * y, 0.0)) for y in (top, bottom))
        if i == STRIPES // 2:
            left = cx - a + PULL * 2 * r
            d = f"M{left:.2f},{cy + top:.2f} h{2 * a:.2f} v{stripe:.2f} h{-2 * a:.2f} Z"
            parts.append(f'<path d="{d}" fill="{accent}"/>')
            continue
        d = (
            f"M{cx - a:.2f},{cy + top:.2f} H{cx + a:.2f} A{r},{r} 0 0 1 {cx + b:.2f},{cy + bottom:.2f} "
            f"H{cx - b:.2f} A{r},{r} 0 0 1 {cx - a:.2f},{cy + top:.2f} Z"
        )
        parts.append(f'<path d="{d}" fill="{fill}"/>')
    return "\n".join(parts)


def svg(width: int, height: int, title: str, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" \
height="{height}" role="img" aria-label="{title}">
<title>{title}</title>
{body}
</svg>
"""


def lockup(x: float, y: float, on: str = "dark", r: float = 44) -> tuple[str, float, float]:
    """The mark and the name side by side from x, centred on y, the name's x-height on the pulled-out
    stripe; with its width and how far the name rises above the mark."""
    gap = 28 * r / 44
    name, name_width, x_height, top = typeset(*NAME[:2], "candor", 88 * r / 44, NAME[2])
    name_x = x + 2 * r * (1 + PULL) + gap
    base = y + x_height / 2
    body = f"""{mark(x + r, y, r, on)}
<path transform="translate({name_x:.2f} {base:.2f})" d="{name}" fill="{ON[on][0]}"/>"""
    return body, name_x + name_width - x, max(y - r - (base + top), 0.0)


def banner() -> str:
    width, height = 1000, 290
    line = "Removes a language model's refusal direction, measured on the file you ship."
    text, text_width, _, _ = typeset(*TEXT, line, 20)
    caps, caps_width, _, _ = typeset(*CAPS, "LLAMA.CPP     QUANTISED GGUF     THINKING MODELS", 11, 0.16)
    _, lockup_width, _ = lockup(0, 0)
    logo, _, _ = lockup((width - lockup_width) / 2, 95)
    body = f"""<rect width="{width}" height="{height}" rx="14" fill="{INK}"/>
{logo}
<rect x="{width / 2 - 24}" y="170" width="48" height="1.5" fill="{MINT}" fill-opacity="0.7"/>
<path transform="translate({(width - text_width) / 2:.2f} 208)" d="{text}" fill="{GREY}"/>
<path transform="translate({(width - caps_width) / 2:.2f} 240)" d="{caps}" fill="{MINT}" \
fill-opacity="0.8"/>"""
    return svg(width, height, "candor: removes a language model's refusal direction", body)


def header() -> str:
    """The logo alone on ink, for the top of a model card, where the viewer's theme can't be known."""
    width, height = 1000, 180
    _, lockup_width, _ = lockup(0, 0, r=36)
    body, _, _ = lockup((width - lockup_width) / 2, height / 2, r=36)
    return svg(
        width, height, "candor", f'<rect width="{width}" height="{height}" rx="14" fill="{INK}"/>\n{body}'
    )


def logo(on: str) -> str:
    """The mark and the name with no background, cropped to them."""
    r = 44
    _, width, rise = lockup(0, 0, on, r)
    body, _, _ = lockup(0, r + rise, on, r)
    return svg(round(width), round(2 * r + rise), "candor", body)


def bare_mark(on: str) -> str:
    """The mark alone with no background, cropped to it."""
    r = 50
    return svg(round(2 * r * (1 + PULL)), 2 * r, "candor", mark(r, r, r, on))


def avatar() -> str:
    """The mark alone on a square, for profile pictures, which platforms crop to a circle."""
    size, r = 512, 150
    body = f'<rect width="{size}" height="{size}" fill="{INK}"/>\n{mark(size / 2 - PULL * r, size / 2, r)}'
    return svg(size, size, "candor", body)


def png(name: str, width: int, height: int) -> None:
    """Renders NAME.svg as NAME.png at twice its size, with its rounded corners transparent."""
    with tempfile.TemporaryDirectory() as folder:
        page, image = Path(folder) / "page.html", (HERE / name).with_suffix(".svg").as_uri()
        page.write_text(
            f'<body style="margin:0;background:transparent"><img src="{image}" width="{width}" '
            f'height="{height}" style="display:block"></body>'
        )
        subprocess.run(
            [
                CHROME,
                "--headless=new",
                "--allow-file-access-from-files",
                "--hide-scrollbars",
                "--default-background-color=00000000",
                "--force-device-scale-factor=2",
                f"--user-data-dir={folder}",
                f"--window-size={width},{height}",
                f"--screenshot={HERE / name}.png",
                page.as_uri(),
            ],
            check=True,
            capture_output=True,
        )


def main() -> None:
    (HERE / "banner.svg").write_text(banner())
    (HERE / "avatar.svg").write_text(avatar())
    (HERE / "header.svg").write_text(header())
    for on in ON:
        (HERE / f"logo-{on}.svg").write_text(logo(on))
        (HERE / f"mark-{on}.svg").write_text(bare_mark(on))
    png("avatar", 512, 512)
    png("header", 1000, 180)
    png("banner", 1000, 290)


if __name__ == "__main__":
    main()
