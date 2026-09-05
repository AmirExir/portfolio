# Portfolio website

The public portfolio is a static site served from the repository root. It presents Amir Exir’s engineering background, project demonstrations, credentials, and downloads. It requires no build step or JavaScript framework.

## File map

| Location | Purpose |
| --- | --- |
| `index.html` | Page content, project links, navigation, SEO metadata, and embedded media |
| `assets/css/portfolio.css` | Colors, typography, layouts, responsive rules, and print styles |
| `assets/js/portfolio.js` | Navigation, selected-work search, galleries, email copying, and document feed rendering |
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

The site uses system fonts and existing photos/screenshots. Content and navigation remain available without JavaScript; galleries then scroll horizontally. Motion is disabled when the visitor requests reduced motion. The mobile AI assistant link appears in the footer to avoid covering page content.

## Presentation refinements

- The cream-and-forest-green palette, existing photography, project text, URLs, and downloads are retained. Project descriptions, metadata, filters, and professional credentials have larger text; search inputs use 16px text to avoid mobile focus zoom.
- The two `work-card--featured` cards (AELab and Grid Atlas) lead the unfiltered collection above 1100px. Supporting cards follow in a four-column row. Filtering restores the regular grid; browsers without CSS `:has()` also retain the regular layout. Keep source order and `hidden` attributes intact.
- Experience uses a chronological rail, with the first listed role visually emphasized. Dates and roles are unchanged. The introductory column stays visible alongside the timeline on wide screens and returns to normal flow on smaller screens and in print.
- Navigation switches to its menu at 820px so intermediate tablet widths do not crowd the desktop links. At 380px and below, hero actions stack into full-width buttons.
- Project cards respond to keyboard focus as well as pointer hover. Gallery arrows and pagination buttons have 44px hit areas. Arrows share the image's grid row, so long pagination lists cannot move them over the page buttons. Galleries remain manual, with reduced-motion and no-JavaScript behavior preserved.

## Validation

Run the local address checks without extra Python dependencies:

```sh
python3 -m unittest discover -s tests -p test_portfolio_site.py -v
node --check assets/js/portfolio.js
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

The browser test serves local files through intercepted requests and blocks external network requests. It checks filtering/search/reset, keyboard and manual galleries, mobile menu behavior, clipboard failure, dated document snapshots, malformed feed responses, and unsafe document URLs. Review layouts at 1440, 1024, 768, 390, and 320 pixels, including with JavaScript disabled. Browser screenshots and temporary profiles should remain outside Git.

## Content boundaries

The engineering descriptions and credentials originate from the existing portfolio. The AI master’s program is explicitly marked as in progress, consistent with the original graduate-student introduction. This design work does not validate professional credential status or introduce new engineering results.

The ERCOT panel shows the snapshot’s generation date and supplied document status/effectiveness notes. A successful fetch does not prove that a document is current or governing. The panel reports missing metadata and load failures instead of presenting a saved feed as live regulatory information. The underlying feed, Atlas datasets, and engineering calculations are unchanged.

Remote Streamlit apps, credential badges, analytics, the music embed, and Atlas map dependencies still depend on their providers and network availability. Offline browser checks validate the portfolio interactions, not those external services.
