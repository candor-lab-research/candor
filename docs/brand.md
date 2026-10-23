# Brand

candor should look like a research institution, not a startup: restrained, precise and measured.
Everything visual made for it (the README and docs, diagrams, model cards, pages, charts, slides)
follows this page.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images/logo-dark.svg">
    <img src="images/logo-light.svg" height="72" alt="candor">
  </picture>
</p>

## The mark

A disc of seven stripes, a representation split into its directions, with the middle one pulled out
along itself: the one direction candor removes. Its files are in `images/`:

| File | Use |
| :--- | :--- |
| `logo-dark.svg`, `logo-light.svg` | The mark and the name, for dark or light backgrounds; no background of their own |
| `mark-dark.svg`, `mark-light.svg` | The mark alone |
| `avatar.svg` | Profile pictures (GitHub, Hugging Face): the mark on ink, safe to crop to a circle |
| `banner.svg` | The README's header |
| `header.svg` | The top of every model card, on its own ink since a card's theme can't be known |
| `avatar.png`, `header.png`, `banner.png` | PNG renders for sites that need them: profile pictures, model cards and the Hugging Face org card |

- Leave clear space around the logo of at least half the disc's width.
- The smallest sizes are 16 px tall for the mark and 24 px for the logo.
- The pulled-out stripe is always the accent and the other stripes the text colour. Never recolour,
  rotate, outline or add effects to the mark, or set the name in another typeface.
- The name is always lowercase, in text and titles too, even at the start of a sentence: candor,
  and the lab is candor lab. Identifiers alone keep the release series' spelling: repository IDs,
  file names and GGUF metadata follow `MODEL-Candor-Uncensored-GGUF`, as every release does, so the
  series stays one pattern and its links keep working.
- Use the files `brand.py` draws; never type the name in a live font as a logo.
- Model cards open with `header.png`, right after their front matter; the org card opens with
  `banner.png`. Each repository carries its own copy.

## Colour

| Role | On dark | On light | Use |
| :--- | :--- | :--- | :--- |
| Background | `#0d1117` | `#ffffff` | Pages and banners (dark is ink) |
| Surface | `#151b23` | `#f6f8fa` | Cards and panels |
| Line | `#3d444d` | `#d1d9e0` | Borders and rules |
| Chip | `#262c36` | `#eaeef2` | Command chips, neutral fills |
| Text | `#e6edf3` | `#1f2328` | Body text (the name and the mark: `#f0f6fc` and `#0d1117`) |
| Muted | `#9198a1` | `#59636e` | Secondary text, arrows, references |
| Accent | `#5fd7af` | `#0a7d5f` | candor itself: the pulled-out stripe, candor's results, one highlight per view |
| Accent soft | `#12352b` | `#d7f4e9` | Fills behind accent text |
| Warning | `#f0883e` | `#bc4c00` | Only what fails or goes wrong |
| Warning soft | `#3d2513` | `#fff1e5` | Fills behind warning text |

- Ink and greys carry everything. The accent is used sparingly, and warning only for failures.
- No gradients, glows, shadows, emoji or pill badges.
- Dark first, with a light version wherever the viewer's theme can change, as GitHub's can
  (`<picture>` with `prefers-color-scheme`).

## Type

| Role | Typeface | Setting |
| :--- | :--- | :--- |
| The name, display headings | Source Serif 4 | Optical size 60, weight 560, tracking -1% |
| Text and titles | Inter | 400 for text, 600-700 for titles |
| Labels | Inter | 500, upper case, 11-12 px, tracking 0.16 em, in the accent or muted |
| Numbers in tables | Inter | Tabular figures (`font-feature-settings: "tnum"`) |
| Code and terminals | `ui-monospace, SFMono-Regular, Menlo, Consolas, monospace` | |

Both typefaces are under the SIL Open Font License, from Google Fonts. Pages load them from Google
Fonts. An SVG shown as an image can't load fonts, so the logo and banner are outlines and diagrams
use the system sans (`-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial,
sans-serif`).

## Layout

- **Headers:** the logo centred, a short accent rule (48 by 1.5 px), one line of text, one line of
  labels. Nothing else.
- **Cards:** 12-14 px corners, a 1.5 px border in the line colour on the surface colour. The card
  that is candor's gets the accent border.
- **Diagrams:** steps numbered in accent-soft circles, arrows in muted grey, text centred in cards.
- **Space:** generous, on a grid; body text left-aligned.

## Charts

- candor's edits in the accent; references (the unedited model, heretic, public edits) in greys;
  warning only for failures, such as a trial over the cap.
- Thin marks, few gridlines, labels on the data rather than legends, tabular figures.

## Voice

Formal, plain and exact. Numbers, never prompt or answer text. No hype, exclamation marks or emoji.

## Making graphics

- `images/brand.py` draws the logo, mark, avatar and banner.
- `images/draw.py` draws the diagrams in these colours, light and dark. New diagrams go there so they
  share its palette and both themes.
- [development.md](development.md#images) has the commands.
