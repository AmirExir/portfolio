## Signal / Field visual assets

The portfolio's cinematic artwork is decorative. It does not show a real study
case, operating conditions, power-flow results, or model performance.

- `assets/images/grid-horizon-concept.jpg`: generated with the built-in imagegen
  tool on September 23, 2026. The original PNG was converted to JPEG at quality 84
  using macOS `sips` for delivery (about 415 KB). No original portrait was edited.
  The rendered page labels the landscape as AI-generated conceptual artwork.
- `AmirinSubstation.jpeg` and the existing `amir-substation-*.webp` files:
  unchanged original documentary photograph of Amir. Shown separately in the
  Field Notes composition; never composited into the generated landscape.
- Project-window images: genuine application screenshots, not generated mockups.
- `assets/fonts/InstrumentSerif-Regular.ttf` and
  `assets/fonts/InstrumentSerif-Italic.ttf`: self-hosted Instrument Serif from
  the Google Fonts repository, distributed under the adjacent
  `InstrumentSerif-OFL.txt`. No runtime font CDN requests.

### AELab gallery refresh

The 20 PNGs in `assets/images/aelab/` were freshly captured from AELab25 on
September 23, 2026 using the application's existing
`Docs/capture_aelab25_screenshots.py` harness. They show the current v25.1
PSS/E / TARA / Utilities shell, including the updated TARA contingency-profile
selection. These are not the older August PNGs in the source catalog.
The website uses byte-for-byte copies of the fresh capture outputs.

The harness captures only the real application window and normalizes it to
1600 × 900. Its IDV library and example directory were isolated in temporary
storage, and its sanitizer cleared local paths and user example presets.
Visible inputs are blank, relative `DEMO/...` paths, or built-in documentation
defaults. Populated planning examples
retain the embedded synthetic-demo labels; empty result tables are intentional.
No licensed analysis was executed to create these captures. Their numeric
defaults and demonstration jobs are interface examples, not study outcomes.
The portfolio import does not further alter pixels, fill tables, or invent
results. The source app, its original screenshot catalog, and user study files
were not modified. Some scrollable panels show only their initial viewport;
the gallery does not claim that every control is visible in one screenshot.

The eight owner-selected original result/visualization slides stay at their
original positions: 2, 4, 10, 23, 25, 26, 27, and 28. Their existing assets are
unchanged. Other positions now cover varied current setup, comparison,
validation, contingency, and TARA workflows instead of repeated legacy screens;
alt text describes the actual replacement image. No engineering result is
inferred from those retained images.

`docs/aelab-gallery.json` records the source repository revision, capture
script and selection, catalog identities, exported filenames, slide order,
file sizes, SHA-256 hashes, and new-image dimensions.
The website and its tests use only the copied local assets; they do not need
the sibling application repository. The existing overview PDF and all legacy
image URLs remain intact.

To reproduce the selected UI-only captures from the source repository, run
the existing harness with a new temporary output directory and isolated
`AELAB_IDV_CATALOG_PATH` / `AELAB_IDV_EXAMPLES_DIR` locations:

```sh
./.venv/bin/python Docs/capture_aelab25_screenshots.py \
  --output-dir <temporary-output-directory> \
  --only 01,02,03,04,06,08,16,20,24,25,26,27,29,30,34,38,40,41,44,46
```

This requires the app's existing Tk/Pillow environment and macOS window-capture
permission. Do not substitute real studies or remove the synthetic-demo labels.

### Final image-generation prompt

```text
Use case: stylized-concept
Asset type: wide cinematic background for a personal power-systems and AI engineering portfolio, not a UI mockup.
Primary request: an epic yet refined cinematic landscape where a real electrical transmission corridor across dark Texas-like rolling terrain visually becomes a luminous network of intelligence. This accompanies an original documentary portrait elsewhere on the page; do not include people.
Scene: blue-hour to night, vast dark undulating terrain seen from elevated viewpoint, realistic lattice transmission towers and gently sagging three-phase conductors receding from the left foreground toward a distant warm horizon slightly right of center. Carefully drawn industrial proportions. The ground subtly carries fine warm amber traces branching like an intelligent network, dissolving into tiny points of light in the distance. Above: deep nearly black petroleum-blue sky, very subtle faint electric-cyan atmospheric haze, no planets. A sophisticated mixture of engineering reality and abstract data, not a fantasy city.
Composition: ultra-wide landscape around 2.4:1, lower half rich with terrain and tower detail, upper third dark negative space. No single giant foreground object. Powerful depth, volumetric haze, tiny warm light sources against ink-black shadows; palette black, desaturated deep teal, champagne amber. Fine photographic detail, premium cinematic art direction, restrained film grain, not cartoon or isometric.
Constraints: no text, labels, logos, frames, interface, numbers, people, robots, lightning bolts, purple neon, floating cubes, fake dashboard. This is conceptual artwork, not an engineering study.
```

### Motion assets

`assets/js/cinematic-fields.js` generates perspective-projected signal fabric,
source/evidence fields, and neural fields in code. It is not video, a captured
simulation, or a representation of a specific electrical network. The illustrations
are independent of the RAG agents and study tools linked from the portfolio.
