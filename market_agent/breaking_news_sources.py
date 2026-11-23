"""Publisher-maintained public market news feeds used by the dashboard and n8n."""

BREAKING_NEWS_FEEDS: tuple[tuple[str, str], ...] = (
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("NVIDIA Newsroom", "https://nvidianews.nvidia.com/releases.xml"),
)
