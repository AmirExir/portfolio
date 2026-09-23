# Portfolio website

The public portfolio is a static site served from the repository root. It presents Amir Exir’s engineering background, project demonstrations, credentials, and downloads. It requires no build step or JavaScript framework.

## File map

| Location | Purpose |
| --- | --- |
| `index.html` | Page content, project links, navigation, SEO metadata, and embedded media |
| `assets/css/portfolio.css` | Colors, typography, layouts, responsive rules, and print styles |
| `assets/css/portfolio-motion.css` | Bounded scene layouts, motion controls, and progressive visual enhancement |
| `assets/css/portfolio-cinematic.css` | Immersive hero, field-story composition, project windows, and large illustration chapters |
| `assets/js/portfolio.js` | Navigation, selected-work search, galleries, email copying, and document feed rendering |
| `assets/js/engineering-scenes.js` | Deterministic canvas renderers for the grid, Atlas, retrieval, neural, forecast, and workflow illustrations |
| `assets/js/cinematic-fields.js` | Larger signal-field, evidence-flow, and learning illustrations |
| `assets/js/portfolio-motion.js` | Shared animation scheduling, viewport visibility, pointer interaction, pause control, and canvas sizing |
| `assets/fonts/` | Self-hosted Instrument Serif font files and their license |
| `assets/images/grid-horizon-concept.jpg` | Labeled, AI-generated conceptual grid landscape in Field Notes |
| `docs/visual-assets.md` | Artwork prompt, provenance, font source, and visual-content boundaries |
| `assets/favicon.svg` | Local browser icon |
| `grid-atlas.html`, `grid-atlas.js` | Existing standalone and embedded Grid Atlas |
| `ERCOTAPI/latest_ercot_updates.json` | Existing document snapshot consumed by the portfolio |
| Root images and PDFs | Existing public media and download addresses |
| `tests/test_portfolio_site.py` | Static local address and fragment validation |
| `tests/test_aelab_ui_download.py` | Existing AELab PDF download/checksum regression |
| `tests/portfolio_browser.cjs` | Optional offline browser interaction checks |

The Python applications, engineering datasets, Atlas data, and document ingestion pipelines retain their existing locations. Published image and PDF filenames also stay in place so external bookmarks and application references continue to work. New website-only media can go in `assets/images/` and new downloads in `assets/documents/`; update the corresponding links when adding them. Do not relocate engineering source data to website asset folders.

## Preview locally

From the repository root:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Open `http://127.0.0.1:8000/`. Use HTTP for previewing: opening `index.html` directly from disk can prevent the document feed and Atlas from fetching their local JSON files. Stop the server with Ctrl+C.

Deployment continues to use the static root and the existing `CNAME`, `robots.txt`, and `sitemap.xml`. No deployment settings or third-party application URLs were changed by the redesign. Update the home page’s `lastmod` in `sitemap.xml` when publishing a substantive content change.

## Update content and addresses

- Edit the six selected-work cards in `index.html` under `#projects`. Each card has `data-project-card` and a space-separated `data-category` using `engineering`, `ai`, or `data`. Search matches the text in these cards. Additional searchable terms can be supplied in `data-search`.
- Search and category filters apply to selected work. The complete project descriptions remain below and are linked through the “Go deeper” navigation. To feature another existing project, add a selected-work card linking to its section or heading ID.
- Maintain the selected card’s image link, title link, and text link together when changing its destination. Internal anchors must match an existing unique `id`.
- Keep paths relative to the repository root in HTML, and encode spaces as `%20`. CSS `url(...)` paths are relative to `assets/css/`; JavaScript feed paths resolve relative to the HTML document. The feed address is configured with `data-feed-url` on `#ercotUpdatesList`.
- Add gallery images inside `.carousel-track` using `.carousel-slide`. Include descriptive alt text. The script generates navigation dots, slide counts, and accessibility state. Galleries advance manually, so users can inspect technical screenshots without a timer.
- The résumé/download disclosure and course certificate disclosure use native HTML `details` elements. They also work without JavaScript. Preserve the AELab overview’s version query and download filename when editing its two existing links; its regression test verifies the published document.
- External links that open another tab use `rel="noopener noreferrer"`. Keep alternative contact methods available if the browser denies clipboard access.

The site combines self-hosted Instrument Serif with system fonts, existing project screenshots, and photography. The immersive hero introduces the engineering themes through an interactive signal field. The original `AmirinSubstation.jpeg` appears in the Field Notes section immediately below the opening material, with responsive WebP copies where supported. That original portrait remains visually separate from the labeled AI-generated grid landscape. Selected-work cards show actual project screenshots in application-window frames; detailed galleries retain their screenshots. Content and navigation remain available without JavaScript; galleries then scroll horizontally. The mobile AI assistant link appears in the footer to avoid covering page content.

## Engineering motion

The page uses seven scene types selected by `.motion-scene[data-scene]`: `field`, `grid`, `atlas`, `evidence`, `rag`, `learning`, and `workflow`. The large `field`, `evidence`, and `learning` scenes belong to `cinematic-fields.js`; the smaller technical insets use `engineering-scenes.js`. The latter also retains its reusable `neural` and `forecast` renderers, although the current page does not place those types. All geometry, motion, and traces are illustrative. They do not represent a study case, measured forecast performance, actual document retrieval, live grid telemetry, or an engineering conclusion.

Both renderers draw projected 3D geometry with the browser's 2D canvas API. This keeps the site independent of WebGL, remote rendering libraries, model downloads, and a build pipeline. The tradeoff is that these are purpose-built illustrations rather than a general 3D viewer. Keep analysis tools and real datasets in their existing applications; never make the decorative renderers a source of engineering values.

The hero's native buttons switch between power systems, knowledge systems, and machine learning. `[data-field-mode]` controls `#heroField`, with a single `aria-pressed` selection and a descriptive live region. Keyboard activation works through the buttons' native Enter/Space behavior. Selecting a mode redraws its illustration even when automatic motion is paused; it does not resume the animation. These controls stay hidden when their canvas cannot initialize or JavaScript is disabled.

`portfolio-motion.js` runs both rendering families through one scheduler, capped at 30 frames per second. Rendering pauses when scenes leave the viewport or the document is hidden. The global “Pause animations” control stops automatic motion; “Play animations” resumes it. A system preference for reduced motion takes priority and displays static scenes. Scene content becomes visible after its first successful draw. Static scene styling, the original portrait, and project screenshots remain available if canvas initialization fails.

Scene hosts have bounded CSS heights and absolutely positioned canvases. Large cinematic scenes have a 900px upper height limit; compact insets have a 360px limit. CSS dimensions determine the backing bitmap, never the reverse. The runtime also applies defensive canvas positioning and host bounds if component stylesheets are missing or stale. Pixel density is capped at 1.5x for cinematic scenes wider than 820px and 2x elsewhere, with each bitmap dimension limited to 2048 pixels. This limits rendering work and prevents the previous Retina resize feedback bug, where an enlarged canvas bitmap increased its parent size and triggered another enlargement. Preserve this separation whenever changing scene markup or styles.

To add a placement, reuse an existing scene type and the page's `.motion-scene` markup with its canvas and accessible surrounding text. Keep scene meaning in visible labels and project descriptions; decorative canvases are hidden from assistive technology. Do not add a separate animation loop for each placement. Update the scene asset version query strings together when publishing changed markup, styles, or rendering code so cached assets remain compatible.

## Presentation refinements

- The dark, immersive opening contrasts with the lighter editorial and project sections. Large serif typography, the original field portrait, and real application screenshots give the animations context. Project URLs, downloads, and factual credentials remain tied to the existing portfolio; search inputs use 16px text to avoid mobile focus zoom.
- The two `work-card--featured` cards (AELab and Grid Atlas) lead the unfiltered collection above 1100px. Supporting cards follow in a four-column row. Filtering restores the regular grid; browsers without CSS `:has()` also retain the regular layout. Keep source order and `hidden` attributes intact.
- Experience uses a chronological rail, with the first listed role visually emphasized. Dates and roles are unchanged. The introductory column stays visible alongside the timeline on wide screens and returns to normal flow on smaller screens and in print.
- Navigation switches to its menu at 820px so intermediate tablet widths do not crowd the desktop links. At 380px and below, the compact header hides its secondary tagline and the two hero actions use tighter padding while keeping 44px targets.
- Project cards respond to keyboard focus as well as pointer hover. Gallery arrows and pagination buttons have 44px hit areas. Arrows share the image's grid row, so long pagination lists cannot move them over the page buttons. Galleries remain manual, with reduced-motion and no-JavaScript behavior preserved.

## Validation

Run the local address checks without extra Python dependencies:

```sh
python3 -m unittest discover -s tests -p test_portfolio_site.py -v
node --check assets/js/portfolio.js
node --check assets/js/engineering-scenes.js
node --check assets/js/cinematic-fields.js
node --check assets/js/portfolio-motion.js
```

If pytest is available, include the existing download regression:

```sh
python3 -m pytest tests/test_portfolio_site.py tests/test_aelab_ui_download.py -q
```

The address checks cover the home page’s local links, images, scripts, stylesheet assets, embedded pages, and fragments, including case-sensitive filenames. They do not crawl remote applications or validate the entire repository.

Optional browser checks require Node.js and Playwright. They are development tools only; the deployed site has no package dependencies. Use an existing Playwright installation, or install one into a temporary directory:

```sh
npm install --prefix /tmp/portfolio-browser-check playwright
/tmp/portfolio-browser-check/node_modules/.bin/playwright install chromium
PORTFOLIO_PLAYWRIGHT_MODULE=/tmp/portfolio-browser-check/node_modules/playwright node tests/portfolio_browser.cjs
```

`PORTFOLIO_PLAYWRIGHT_MODULE` can point to another installed Playwright module. Set `PORTFOLIO_BROWSER_EXECUTABLE` to use an existing Chrome/Chromium executable instead of Playwright’s downloaded browser.

The browser test serves local files through intercepted requests and blocks external network requests. It checks all seven placed scene types, keyboard hero-mode selection, animation and global pause/resume across both rendering families, offscreen and document-visibility suspension, static reduced-motion rendering, and original portrait/project-image fallbacks. The visibility test dispatches a visibility-change event with simulated hidden state because operating-system background-tab behavior is unreliable in a headless browser. Layout checks cover 320–1440 pixels and both 1x and 2x pixel density. A separate 2x context deliberately fails the motion component stylesheets and checks repeated resizes for bounded canvas and document dimensions. Another context makes scene canvases return no 2D context, verifying that the portfolio's content, navigation, and filtering continue to work.

Existing regression checks also cover filtering/search/reset, keyboard and manual galleries, mobile navigation, clipboard failure, dated document snapshots, malformed feed responses, and unsafe document URLs. Inspect representative desktop and mobile screenshots as well: functional checks cannot establish visual quality. Browser screenshots and temporary profiles should remain outside Git.

## Content boundaries

The engineering descriptions and credentials originate from the existing portfolio. Based on the owner-provided status, the AI master’s program is described as a graduating degree candidate rather than a conferred degree. Update that language after formal conferral. This design work does not independently validate professional credential status or introduce new engineering results.

The ERCOT panel shows the snapshot’s generation date and supplied document status/effectiveness notes. A successful fetch does not prove that a document is current or governing. The panel reports missing metadata and load failures instead of presenting a saved feed as live regulatory information. The underlying feed, Atlas datasets, and engineering calculations are unchanged.

Remote Streamlit apps, credential badges, analytics, the music embed, and Atlas map dependencies still depend on their providers and network availability. Offline browser checks validate the portfolio interactions, not those external services.
