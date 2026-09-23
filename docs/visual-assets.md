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
- Project-window images: existing repository screenshots, not generated mockups.
- `assets/fonts/InstrumentSerif-Regular.ttf` and
  `assets/fonts/InstrumentSerif-Italic.ttf`: self-hosted Instrument Serif from
  the Google Fonts repository, distributed under the adjacent
  `InstrumentSerif-OFL.txt`. No runtime font CDN requests.

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
