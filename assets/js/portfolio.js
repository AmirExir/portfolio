(() => {
  'use strict';

  function initializeNavigation() {
    const toggle = document.getElementById('menuToggle');
    const links = document.getElementById('navLinks');
    if (!toggle || !links) return;

    const setOpen = (isOpen, restoreFocus = false) => {
      links.classList.toggle('is-open', isOpen);
      toggle.setAttribute('aria-expanded', String(isOpen));
      toggle.setAttribute('aria-label', isOpen ? 'Close navigation' : 'Open navigation');
      if (restoreFocus) toggle.focus();
    };

    toggle.addEventListener('click', () => {
      setOpen(toggle.getAttribute('aria-expanded') !== 'true');
    });
    links.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => setOpen(false));
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && toggle.getAttribute('aria-expanded') === 'true') {
        setOpen(false, true);
      }
    });
    document.addEventListener('click', (event) => {
      if (!links.contains(event.target) && !toggle.contains(event.target)) setOpen(false);
    });
    document.addEventListener('focusin', (event) => {
      if (!links.contains(event.target) && !toggle.contains(event.target)) setOpen(false);
    });
  }

  function initializeCarousels() {
    document.querySelectorAll('[data-carousel]').forEach((carousel) => {
      const track = carousel.querySelector('.carousel-track');
      const slides = Array.from(carousel.querySelectorAll('.carousel-slide'));
      const previous = carousel.querySelector('[data-prev]');
      const next = carousel.querySelector('[data-next]');
      const dots = carousel.querySelector('[data-dots]');
      if (!track || !slides.length) return;

      let index = 0;
      carousel.tabIndex = 0;
      carousel.setAttribute('role', 'group');
      carousel.setAttribute('aria-roledescription', 'carousel');

      const status = document.createElement('p');
      status.className = 'carousel-status';
      status.setAttribute('role', 'status');
      status.setAttribute('aria-live', 'polite');
      status.setAttribute('aria-atomic', 'true');
      carousel.appendChild(status);

      slides.forEach((slide, slideIndex) => {
        slide.setAttribute('role', 'group');
        slide.setAttribute('aria-roledescription', 'slide');
        slide.setAttribute('aria-label', `${slideIndex + 1} of ${slides.length}`);
      });

      const update = (nextIndex) => {
        const destination = (nextIndex + slides.length) % slides.length;
        if (destination !== index && slides[index].contains(document.activeElement)) {
          carousel.focus({ preventScroll: true });
        }
        index = destination;
        track.style.transform = `translateX(${-index * 100}%)`;
        slides.forEach((slide, slideIndex) => {
          const inactive = slideIndex !== index;
          slide.inert = inactive;
          slide.setAttribute('aria-hidden', String(inactive));
          if (inactive) slide.querySelectorAll('video, audio').forEach((media) => media.pause());
        });
        if (dots) {
          Array.from(dots.children).forEach((dot, dotIndex) => {
            dot.setAttribute('aria-current', String(dotIndex === index));
          });
        }
        status.textContent = `Slide ${index + 1} of ${slides.length}`;
      };

      if (dots) {
        dots.replaceChildren();
        slides.forEach((slide, dotIndex) => {
          const dot = document.createElement('button');
          dot.type = 'button';
          dot.className = 'carousel-dot';
          const caption = slide.querySelector('img')?.alt;
          dot.setAttribute('aria-label', `Show slide ${dotIndex + 1}${caption ? `: ${caption}` : ''}`);
          dot.addEventListener('click', () => update(dotIndex));
          dots.appendChild(dot);
        });
        dots.hidden = slides.length === 1;
      }

      if (previous) {
        previous.hidden = slides.length === 1;
        previous.addEventListener('click', () => update(index - 1));
      }
      if (next) {
        next.hidden = slides.length === 1;
        next.addEventListener('click', () => update(index + 1));
      }
      carousel.addEventListener('keydown', (event) => {
        if (event.target.closest('input, textarea, select, video, audio, [contenteditable]')) return;
        if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
          event.preventDefault();
          update(index + (event.key === 'ArrowLeft' ? -1 : 1));
        }
      });
      update(0);
    });
  }

  function initializeProjectSearch() {
    const form = document.getElementById('siteSearch');
    const input = document.getElementById('searchInput');
    const cards = Array.from(document.querySelectorAll('[data-project-card]'));
    if (!form || !input || !cards.length) return;

    const filters = Array.from(document.querySelectorAll('[data-project-filter]'));
    const status = document.getElementById('searchStatus');
    const count = document.getElementById('projectCount');
    const empty = document.getElementById('projectEmpty');
    const clear = document.getElementById('clearSearch');
    let category = 'all';
    const searchable = cards.map((card) => ({
      card,
      text: `${card.textContent} ${card.dataset.search || ''}`.toLocaleLowerCase(),
      categories: (card.dataset.category || '').split(/\s+/),
    }));

    const update = () => {
      const query = input.value.trim().toLocaleLowerCase();
      const words = query.split(/\s+/).filter(Boolean);
      let visibleCount = 0;
      searchable.forEach(({ card, text, categories }) => {
        const matches = (category === 'all' || categories.includes(category))
          && words.every((word) => text.includes(word));
        card.hidden = !matches;
        if (matches) visibleCount += 1;
      });
      filters.forEach((button) => {
        button.setAttribute('aria-pressed', String(button.dataset.projectFilter === category));
      });
      const totalText = `${visibleCount} of ${cards.length} projects`;
      if (count) count.textContent = `${visibleCount} ${visibleCount === 1 ? 'project' : 'projects'}`;
      if (status) {
        status.textContent = visibleCount
          ? `${totalText} shown${query ? ` matching “${input.value.trim()}”` : ''}.`
          : 'No projects match these filters. Try a different search or clear the filters.';
      }
      if (empty) empty.hidden = visibleCount !== 0;
      if (clear) clear.hidden = !query && category === 'all';
    };

    input.addEventListener('input', update);
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      update();
    });
    filters.forEach((button) => {
      button.addEventListener('click', () => {
        category = button.dataset.projectFilter || 'all';
        update();
      });
    });
    clear?.addEventListener('click', () => {
      input.value = '';
      category = 'all';
      update();
      input.focus({ preventScroll: true });
    });
    update();
    document.querySelectorAll('[data-project-tools]').forEach((element) => {
      element.hidden = false;
    });
  }

  function initializeEmailCopy() {
    document.querySelectorAll('[data-copy-email]').forEach((button) => {
      button.addEventListener('click', async () => {
        const email = button.dataset.copyEmail;
        const status = document.getElementById('copyEmailStatus');
        if (!email) return;
        try {
          await navigator.clipboard.writeText(email);
          if (status) status.textContent = `${email} copied to clipboard.`;
        } catch {
          if (status) status.textContent = `Copy is unavailable in this browser. Select and copy: ${email}`;
        }
      });
    });
  }

  const textValue = (value) => typeof value === 'string' ? value.trim() : '';
  const isRecord = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);

  function safeDocumentUrl(value) {
    if (!textValue(value)) return null;
    try {
      const url = new URL(value);
      return ['https:', 'http:'].includes(url.protocol) && !url.username && !url.password
        ? url.href : null;
    } catch {
      return null;
    }
  }

  function renderFeedItem(item, revisionIssues) {
    const revision = revisionIssues[textValue(item.revision_id)];
    const detail = isRecord(revision) ? revision : {};
    const label = textValue(item.document_number) || textValue(item.title);
    const title = textValue(item.issue_title) || textValue(detail.issue_title);
    const sources = Array.isArray(item.sources) ? item.sources.map(textValue).filter(Boolean) : [];
    const source = sources.join(', ') || textValue(item.source) || 'ERCOT';
    const url = safeDocumentUrl(item.url);
    const entry = document.createElement('li');
    const heading = document.createElement(url ? 'a' : 'strong');
    heading.textContent = `${label}${title && title !== label ? ` — ${title}` : ''}`;
    if (url) {
      heading.href = url;
      heading.target = '_blank';
      heading.rel = 'noopener noreferrer';
    }
    entry.appendChild(heading);

    const metadata = document.createElement('p');
    metadata.className = 'update-meta';
    const published = textValue(item.published_date) || textValue(detail.date_posted);
    const documentStatus = textValue(item.status) || textValue(detail.status) || 'Not established';
    metadata.textContent = `${source} · ${published ? `Published: ${published}` : 'Publication date not supplied'} · Status: ${documentStatus}`;
    entry.appendChild(metadata);

    const explanation = textValue(item.explanation);
    if (explanation) {
      const summary = document.createElement('p');
      summary.className = 'update-summary';
      summary.textContent = explanation;
      entry.appendChild(summary);
    }
    const effectiveness = document.createElement('p');
    effectiveness.className = 'update-effectiveness';
    effectiveness.textContent = textValue(item.effectiveness_note) || textValue(detail.effectiveness_note)
      || 'Effectiveness is not established by this feed. Verify the current controlling document before relying on this material.';
    entry.appendChild(effectiveness);
    return entry;
  }

  async function initializeErcotFeed() {
    const list = document.getElementById('ercotUpdatesList');
    const status = document.getElementById('ercotUpdatesStatus');
    if (!list || !status) return;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 10000);
    list.setAttribute('aria-busy', 'true');
    try {
      const response = await fetch(list.dataset.feedUrl || 'ERCOTAPI/latest_ercot_updates.json', {
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`Feed request failed with HTTP ${response.status}`);
      const feed = await response.json();
      if (!isRecord(feed) || !Array.isArray(feed.items)) throw new Error('Invalid document feed');
      if (!feed.items.every((item) => isRecord(item) && (textValue(item.document_number) || textValue(item.title)))) {
        throw new Error('Invalid document entry');
      }
      const revisionIssues = isRecord(feed.revision_issues) ? feed.revision_issues : {};
      const items = feed.items.slice(0, 12);
      const generatedAt = new Date(textValue(feed.generated_at));
      const snapshot = Number.isNaN(generatedAt.getTime())
        ? 'Snapshot generation date unavailable.'
        : `Snapshot generated ${generatedAt.toLocaleDateString('en-US', {
          year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC',
        })} (UTC).`;
      list.replaceChildren(...items.map((item) => renderFeedItem(item, revisionIssues)));
      status.textContent = items.length
        ? `${snapshot} Showing ${items.length} of ${feed.items.length} indexed documents.`
        : `${snapshot} No documents are listed in this snapshot.`;
    } catch {
      list.replaceChildren();
      status.textContent = 'The document feed could not be loaded. Reload the page or open the ERCOT dashboard to browse documents.';
    } finally {
      window.clearTimeout(timeout);
      list.setAttribute('aria-busy', 'false');
    }
  }

  initializeNavigation();
  initializeCarousels();
  initializeProjectSearch();
  initializeEmailCopy();
  initializeErcotFeed();
  document.documentElement.classList.add('js');
})();
