# Portfolio website

The public portfolio is a static site served from the repository root. It presents Amir Exir’s engineering background, project demonstrations, credentials, and downloads. It requires no build step or JavaScript framework.

## File map

| Location | Purpose |
| --- | --- |
| `index.html` | Page content, project links, navigation, SEO metadata, and embedded media |
| `assets/css/portfolio.css` | Colors, typography, layouts, responsive rules, and print styles |
| `assets/css/portfolio-motion.css` | Bounded scene layouts, motion controls, and progressive visual enhancement |
| `assets/css/portfolio-cinematic.css` | Immersive hero, field-story composition, project windows, and large illustration chapters |
| `assets/css/project-stories.css` | Bounded project-story artwork, screenshot fallbacks, phase labels, and responsive stage controls |
| `assets/js/portfolio.js` | Navigation, selected-work search, galleries, email copying, and document feed rendering |
| `assets/js/engineering-scenes.js` | Deterministic canvas renderers for the grid, Atlas, retrieval, neural, forecast, and workflow illustrations |
| `assets/js/cinematic-fields.js` | Larger signal-field, evidence-flow, and learning illustrations |
| `assets/js/power-journey.js` | Conceptual generation, storage, transmission, distribution, and load artwork for the hero |
| `assets/js/project-scene-kit.js` | Shared projection, geometry, line, glow, and label drawing helpers for project stories |
| `assets/js/contingency-scene.js` | Staged illustrative branch opening and alternate-path flow for AELab |
| `assets/js/fault-scene.js` | Synthetic signal, feature-extraction, and classification artwork |
| `assets/js/forecast-scene.js` | Illustrative demand landscape, forecast-origin plane, and future contours |
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
| `tests/project_scene_states.cjs` | Dependency-free deterministic stage and boundary checks for the three project stories |
| `tests/power_journey_geometry.cjs` | Dependency-free generation/storage geometry, flow-direction, and canvas-state checks |

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
- Maintain the three academic entries in `#education` separately from professional credentials. The section uses native fragment links and readable HTML, so the timeline does not depend on animation or JavaScript. Preserve the recorded institution names, degree types, study dates, and the AI degree's candidate status unless the owner provides updated information.

The site combines self-hosted Instrument Serif with system fonts, existing project screenshots, and photography. The immersive hero introduces the engineering themes through an interactive signal field. The original `AmirinSubstation.jpeg` appears in the Field Notes section immediately below the opening material, with responsive WebP copies where supported. That original portrait remains visually separate from the labeled AI-generated grid landscape. Selected-work cards show actual project screenshots in application-window frames; detailed galleries retain their screenshots. Content and navigation remain available without JavaScript; galleries then scroll horizontally. The mobile AI assistant link appears in the footer to avoid covering page content.

## Engineering motion

The page uses nine scene types selected by `.motion-scene[data-scene]`: `field`, `contingency`, `atlas`, `evidence`, `rag`, `learning`, `forecast`, `fault`, and `workflow`. The large `field`, `evidence`, and `learning` scenes belong to `cinematic-fields.js`, which delegates the hero's power-systems mode to `power-journey.js`. The three project stories have their own renderers, sharing the small drawing helpers in `project-scene-kit.js`; the remaining compact technical insets use `engineering-scenes.js`. The controller reserves the `contingency`, `fault`, and `forecast` types for their project renderers. It must not silently use the older compact forecast illustration if the new forecast module is unavailable. All geometry, motion, and traces are illustrative. They do not represent a study case, measured forecast performance, actual document retrieval, live grid telemetry, or an engineering conclusion.

The renderers draw projected 3D geometry with the browser's 2D canvas API. This keeps the site independent of WebGL, remote rendering libraries, model downloads, and a build pipeline. The tradeoff is that these are purpose-built illustrations rather than a general 3D viewer. Keep analysis tools and real datasets in their existing applications; never make the decorative renderers a source of engineering values.

The power journey illustrates six stages: generation and storage, step-up transformer, transmission, step-down substation, distribution, and loads. Its opening equipment represents nuclear generation, gas generation, a hydroelectric dam, a wind farm, a solar farm, and battery storage. `[data-power-sources]` names these six technologies inside the stage legend. Battery storage represents energy storage and exchange, not a primary energy source. These are schematic equipment categories, not a proposed plant design, a particular utility's generation mix, or a physical site shared by every technology. The load destinations represent homes, data centers, crypto mining, industrial facilities, and commercial buildings. `[data-power-legend]` identifies the stages, and the power-mode description names the five load types. These labels appear only while the corresponding artwork is available and selected; AI modes use their own descriptions.

Solar and battery equipment have separate DC pairs entering converters, with distinct AC terminals connecting onward. Storage artwork cycles through discharge, idle, charge, and idle over 18.4 seconds. Discharge pulses move from DC storage through the converter toward the AC collector; charging reverses both the segment order and the packet/trail direction. Idle stops storage transfer without stopping the rest of the grid illustration. The cabinet's nonnumeric level indicator decreases during discharge and increases during charging. This is an animation convention, not a measured state of charge, capacity, charging schedule, efficiency, or dispatch result. Wind rotors and hydro spillway effects are likewise decorative motion rather than calculated generation.

Luminous pulses indicate conceptual energy flow. They do not depict electrons traveling from a generator to a load through an AC system, and no power flow, voltage, current, protection operation, or equipment rating is calculated. The step-down and distribution geometry communicates the electrical stages schematically, rather than specifying actual voltage levels or a utility's service topology. Keep these engineering limitations in the documentation; the on-page hero uses stage and destination labels without simulation or electron-motion warnings.

If `power-journey.js` fails to load or throws while drawing, the hero falls back to its original abstract grid field, hides the stage legend, restores the general field description, and keeps the knowledge and learning modes available. A draw failure is logged once and disables further calls to that optional renderer for the page session.

The hero's native buttons switch between power systems, knowledge systems, and machine learning. `[data-field-mode]` controls `#heroField`, with a single `aria-pressed` selection and a descriptive live region. Keyboard activation works through the buttons' native Enter/Space behavior. Selecting a mode redraws its illustration even when automatic motion is paused; it does not resume the animation. These controls stay hidden when their canvas cannot initialize or JavaScript is disabled.

The AELab, fault-detection, and forecasting project stories use native `.scene-controls` buttons for “Sequence” (the `auto` state) and three manually selected stages. Both the scene host and its buttons use `data-scene-state`; scope button queries to `.scene-controls`. Each button targets its canvas host through `aria-controls`, and a single `aria-pressed` selection identifies the active choice. Selecting a manual stage repaints immediately, including while globally paused or under reduced motion, without restarting automatic motion. `[data-scene-phase]` describes the active visual stage with `aria-live="off"` so looping artwork does not repeatedly interrupt assistive technology. Controls appear only after a successful draw and disappear if that renderer fails. Story descriptions, project links, and screenshots remain available without JavaScript, canvas, or optional renderer assets.

These story stages explain methods rather than report results. The AELab illustration shows a base network, an opened branch, alternate-path flow, and restoration during Auto playback; it does not solve a contingency or establish ratings or violations. The fault story uses synthetic traces to illustrate a signal window, feature extraction, and classification, without claiming a measured fault, classifier accuracy, or protection response time. The forecasting story separates a fixed historical demand landscape, forecast-origin plane, and alternative future contours. Its historical geometry does not change with animation time, and its future contours are neither a trained model's prediction nor a calibrated uncertainty interval. Animation duration is presentation timing, not engineering event time. Preserve these distinctions when updating the artwork or captions.

Each project renderer exposes `draw(ctx, options)`, `getStage({time, state})`, and a `labels` map. Their 12-second Auto cycles are deterministic; manual states hold their named stage while time may continue driving decorative pulses. Invalid state names use Auto, and negative or nonfinite times normalize to zero. Renderers do not create timers, modify the DOM, or resize canvases. Keep the phase resolver shared between its labels and drawing behavior so a selected label cannot drift from the displayed illustration.

`portfolio-motion.js` runs all renderers through one scheduler, capped at 30 frames per second. Rendering pauses when scenes leave the viewport or the document is hidden. The global “Pause animations” control stops automatic motion; “Play animations” resumes it. A system preference for reduced motion takes priority and displays static scenes. Scene content becomes visible after its first successful draw. Static scene styling, the original portrait, and project screenshots remain available if canvas initialization fails.

Scene hosts have bounded CSS heights and absolutely positioned canvases. The hero's runtime height limit is 1000px, accommodating the 980px mobile composition for its generation/storage district; other cinematic scenes retain a 900px limit and compact insets a 360px limit. CSS dimensions determine the backing bitmap, never the reverse. The runtime also applies defensive canvas positioning and host bounds if component stylesheets are missing or stale. Pixel density is capped at 1.5x for cinematic scenes wider than 820px and 2x elsewhere, with each bitmap dimension limited to 2048 pixels. This limits rendering work and prevents the previous Retina resize feedback bug, where an enlarged canvas bitmap increased its parent size and triggered another enlargement. Preserve this separation whenever changing scene markup or styles.

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
node --check assets/js/power-journey.js
node --check assets/js/project-scene-kit.js
node --check assets/js/contingency-scene.js
node --check assets/js/fault-scene.js
node --check assets/js/forecast-scene.js
node --check assets/js/portfolio-motion.js
node tests/project_scene_states.cjs
node tests/power_journey_geometry.cjs
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

The browser test serves local files through intercepted requests and blocks external network requests. It checks all nine placed scene types, keyboard hero-mode and project-stage selection, distinct manual-stage frames, automatic animation and global pause/resume across all renderers, offscreen and document-visibility suspension, static reduced-motion rendering, and original portrait/project-image fallbacks. Manual project selection and Auto must not resume a globally paused scene. Project phase labels must match the selected stage, and controls must retain 44px targets on smaller screens. The separate Node state test locks each project's cycle boundaries, manual states, readable phase labels, default state, and invalid-input behavior without starting a browser or loading datasets. These are UI state tests, not engineering validation.

The power journey's six-stage order, six generation/storage labels, and five load labels are checked in grid mode, including unclipped captions on smaller viewports. AI modes must replace those annotations, and the hero must omit simulation and electron-motion warnings. The dependency-free power-journey test records actual canvas commands to verify repeatable geometry, finite drawing values, changing animated frames, pointer extremes, and balanced canvas state at desktop/mobile dimensions and after an injected drawing failure.

That test instruments private helpers only inside its VM copy; the deployed renderer still exposes only `draw()`. It verifies that the desktop, mobile, and narrow-phone layouts construct all six technologies with distinct shapes after removing translation, scale, color, and captions. It also checks separate AC terminals and DC pairs, storage-state boundaries, actual packet/trail reversal, AC/DC transfer ordering, and idle behavior. The ordering test observes pulses on the real model segments rather than repeating the renderer's schedule formula. These tests cover illustrative geometry and animation logic, not engineering calculations. They complement visual inspection and do not certify equipment designs or validate a power-system model.

A context with `power-journey.js` blocked verifies the original field fallback and working AI controls. Another injects a draw failure after a successful power frame, verifying that the legend updates, fallback animation continues, AI modes remain distinct, and one warning reports the failure. A separate 2x context deliberately fails the motion component stylesheets and checks repeated resizes for bounded canvas and document dimensions. Another makes scene canvases return no 2D context, verifying that content, navigation, and filtering continue to work.

Additional contexts block the three project renderers or their shared drawing kit, verifying hidden unavailable controls, preserved story headings, screenshots and project links, and no substitution of the old compact forecast renderer. Conversely, blocking both older renderer families must leave all three project stories and their controls functional, while the hero retains its static styling and copy. A late injected fault-renderer failure must restore its screenshot, hide only its controls, warn once, and leave neighboring project scenes and the hero working. The same content-preservation checks run without JavaScript and without canvas contexts.

Layout checks cover 320–1440 pixels, including both sides of the 480px/640px source-artwork breakpoints, a shorter 1440×800 desktop viewport, and both 1x and 2x pixel density. The visibility test dispatches a visibility-change event with simulated hidden state because operating-system background-tab behavior is unreliable in a headless browser.

Existing regression checks also cover the restored education timeline and native anchors, filtering/search/reset, keyboard and manual galleries, mobile navigation, clipboard failure, dated document snapshots, malformed feed responses, and unsafe document URLs. Education entries are checked for their source-backed dates, current candidate status, responsive bounds, and availability without JavaScript. Inspect representative desktop and mobile screenshots as well: functional checks cannot establish visual quality. Browser screenshots and temporary profiles should remain outside Git.

## Content boundaries

The engineering descriptions and credentials originate from the existing portfolio. Based on the owner-provided status, the AI master’s program is described as a graduating degree candidate rather than a conferred degree. Update that language after formal conferral. This design work does not independently validate professional credential status or introduce new engineering results.

The education timeline restores UT Austin (M.S. in Artificial Intelligence, August 2024–present), Lamar University (M.Eng. in Electrical and Computer Engineering, January 2019–May 2020), and Shahid Beheshti University (B.S. in Electrical and Computer Engineering, October 2012–July 2017). Institution names and study periods come from the repository résumé and earlier portfolio content. Conflicting GPA values in those sources are not displayed.

The ERCOT panel shows the snapshot’s generation date and supplied document status/effectiveness notes. A successful fetch does not prove that a document is current or governing. The panel reports missing metadata and load failures instead of presenting a saved feed as live regulatory information. The underlying feed, Atlas datasets, and engineering calculations are unchanged.

Remote Streamlit apps, credential badges, analytics, the music embed, and Atlas map dependencies still depend on their providers and network availability. Offline browser checks validate the portfolio interactions, not those external services.
