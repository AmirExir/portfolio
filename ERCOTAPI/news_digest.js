// n8n Code node: Build ERCOT News Digest, runOnceForAllItems.
// State is committed by n8n after a successful production execution. The first
// scheduled run records the current feed without replaying it after deployment.
const MAX_ARTICLE_AGE_HOURS = 36;
const MAX_DIGEST_ARTICLES = 8;
const SEEN_RETENTION_DAYS = 45;
const STATE_VERSION = 1;
const STATE_KEY = 'ercotNewsDigest';

const now = Date.now();
const freshAfter = now - MAX_ARTICLE_AGE_HOURS * 60 * 60 * 1000;
const context = $('ERCOT Request Context').first().json;
const isScheduled = !context.source || context.source === 'schedule';
const payloads = $input.all().map((item) => item.json);
if (!payloads.length || payloads.some((payload) => (
  !payload || payload.error || payload.status !== 'ok' || !Array.isArray(payload.articles)
))) {
  // Do not publish placeholders or expose API error details, which can include
  // request URLs or credentials. A failed feed must remain a failed execution.
  throw new Error('ERCOT news retrieval failed: expected a successful NewsAPI article response.');
}

function cleanText(value) {
  return typeof value === 'string' ? value.replace(/\s+/g, ' ').trim() : '';
}

function titleIdentity(title) {
  return title.normalize('NFKC').toLowerCase()
    .replace(/&amp;/g, ' and ').replace(/&/g, ' and ')
    .replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
}

function canonicalUrl(value) {
  // n8n's sandbox does not expose URL/URLSearchParams. Preserve substantive
  // query parameters (such as an article id), dropping only known tracking keys.
  const url = cleanText(value);
  const parts = /^(https?):\/\/([^/?#\s]+)([^?#\s]*)(?:\?([^#\s]*))?(?:#[^\s]*)?$/i.exec(url);
  if (!parts || parts[2].includes('@')) return null;
  const host = parts[2].toLowerCase().replace(/^www\./, '')
    .replace(parts[1].toLowerCase() === 'https' ? /:443$/ : /:80$/, '');
  if (!/^[a-z0-9.-]+(?::\d+)?$/.test(host)) return null;
  const path = parts[3].replace(/\/+$/, '') || '/';
  const tracking = /^(utm_.*|fbclid|gclid|dclid|msclkid|mc_cid|mc_eid|igshid|_ga|_gl)$/i;
  const query = [];
  for (const parameter of (parts[4] || '').split('&')) {
    if (!parameter) continue;
    let key;
    try {
      key = decodeURIComponent(parameter.split('=', 1)[0].replace(/\+/g, ' '));
    } catch {
      return null;
    }
    if (!tracking.test(key)) query.push(parameter);
  }
  query.sort();
  return host + path + (query.length ? '?' + query.join('&') : '');
}

function isRelevant(article) {
  const text = [article.title, article.description, article.content].map(cleanText).join(' ');
  const ercot = /\b(ercot|electric reliability council of texas)\b/i;
  const texas = /\b(texas|texan)\b/i;
  const grid = /\b(power grid|electric(?:al)? grid|electricity|transmission|interconnection|large loads?|load growth|data cent(?:er|re)s?|substations?|congestion|generation|power plants?|utilities|utility|energy markets?|battery storage)\b/i;
  return ercot.test(text) || (texas.test(text) && grid.test(text));
}

const candidates = [];
for (const article of payloads.flatMap((payload) => payload.articles)) {
  if (!article || typeof article !== 'object') continue;
  const title = cleanText(article.title);
  const urlKey = canonicalUrl(article.url);
  const publishedAt = cleanText(article.publishedAt);
  // Require an actual source timestamp, including timezone; never substitute
  // the execution date for missing publication metadata or accept future news.
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/i.test(publishedAt)) continue;
  const publishedMs = Date.parse(publishedAt);
  if (!title || /^\[removed\]$/i.test(title) || !urlKey || !isRelevant(article)
      || !Number.isFinite(publishedMs) || publishedMs < freshAfter || publishedMs > now) continue;
  const titleKey = titleIdentity(title);
  if (!titleKey) continue;
  candidates.push({
    title,
    url: cleanText(article.url),
    source: cleanText(article.source?.name),
    publishedAt,
    publishedMs,
    description: cleanText(article.description).slice(0, 260),
    identities: ['url:' + urlKey, 'title:' + titleKey],
  });
}
candidates.sort((left, right) => right.publishedMs - left.publishedMs);

// Explicit requests can repeat the current digest without consuming stories
// from the scheduled publication stream or initializing its baseline.
let state;
let previousIdentities = {};
let isBaseline = false;
if (isScheduled) {
  const staticData = $getWorkflowStaticData('node');
  state = staticData[STATE_KEY];
  if (state && (state.version !== STATE_VERSION || !state.seen
      || typeof state.seen !== 'object' || Array.isArray(state.seen)
      || !Number.isFinite(Date.parse(state.initializedAt)))) {
    throw new Error('ERCOT news deduplication state is invalid; restore its saved state before publishing.');
  }
  if (!state) {
    state = { version: STATE_VERSION, initializedAt: new Date(now).toISOString(), seen: {} };
    staticData[STATE_KEY] = state;
    isBaseline = true;
  }
  const retainAfter = now - SEEN_RETENTION_DAYS * 24 * 60 * 60 * 1000;
  for (const [identity, lastSeenAt] of Object.entries(state.seen)) {
    if (!Number.isFinite(lastSeenAt)) {
      throw new Error('ERCOT news deduplication history is invalid; restore its saved state before publishing.');
    }
    if (lastSeenAt < retainAfter) delete state.seen[identity];
  }
  previousIdentities = { ...state.seen };
}

const batchSeen = new Set();
const selected = [];
for (const article of candidates) {
  const alreadySeen = article.identities.some((identity) => (
    Object.prototype.hasOwnProperty.call(previousIdentities, identity) || batchSeen.has(identity)
  ));
  for (const identity of article.identities) {
    batchSeen.add(identity);
    if (state) state.seen[identity] = now;
  }
  // Record every eligible item, including aliases and items beyond the display
  // limit, so an unchanged feed cannot trickle into repeated follow-up posts.
  if (!alreadySeen && selected.length < MAX_DIGEST_ARTICLES) selected.push(article);
}
if (isScheduled && (isBaseline || !selected.length)) return [];

const lines = selected.map((article, index) => {
  const meta = [article.source, article.publishedAt.slice(0, 10)].filter(Boolean).join(' • ');
  return String(index + 1) + '. ' + article.title
    + (meta ? '\n' + meta : '')
    + (article.description ? '\n' + article.description : '')
    + '\n' + article.url;
});
const summary = lines.length ? lines.join('\n\n')
  : 'No current ERCOT or Texas grid news articles were returned.';
return [{ json: {
  message: { content: summary },
  article_count: selected.length,
  publish_to_channel: isScheduled,
} }];
