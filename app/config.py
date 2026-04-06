# ════════════════════════════════════════════════════════════════
#  Central configuration for the Local Research Agent.
#
#  Two tiers:
#    1. This file — all defaults, imported by services/reporting/pipeline.
#    2. Settings model (schemas.py, persisted in DB) — user-overridable
#       subset. When a Setting exists, it takes precedence.
# ════════════════════════════════════════════════════════════════


# ── LLM ────────────────────────────────────────────────────────
OLLAMA_DEFAULT_TIMEOUT = 120
DEFAULT_MAX_TOKENS = 4096
REASONING_TEMPERATURE = 0.4
SYNTHESIS_TEMPERATURE = 0.4
QUERY_GEN_TEMPERATURE = 0.5
GAP_QUERY_TEMPERATURE = 0.6
COORDINATOR_TEMPERATURE = 0.3
ANALYST_TEMPERATURE = 0.3
GOAL_TEMPERATURE = 0.3
OLLAMA_RETRIES = 2

# ── Search ─────────────────────────────────────────────────────
SEARCH_RETRY_COUNT = 5
SEARCH_BACKOFF_BASE = 4.0
SEARCH_BACKOFF_MAX = 30.0
INTER_QUERY_DELAY = 4.0
INTER_ITERATION_COOLDOWN = 5.0

# ── Synthesis ──────────────────────────────────────────────────
TOP_SOURCES_CAP = 9
ANALYST_BATCH_SIZE = 3
SOURCE_CONTENT_LENGTH = 3000
SYNTHESIS_TIMEOUT = 600
ANALYST_TIMEOUT = 300
COORDINATOR_TIMEOUT = 60
SYNTHESIS_MAX_TOKENS = 8192
ANALYST_MAX_TOKENS = 4096
DIRECT_SOURCE_THRESHOLD = 8

# ── Quality scoring ────────────────────────────────────────────
QUALITY_SCORE_GOOD = 5
QUALITY_SCORE_MEDIUM = 3
MIN_TEXT_LENGTH_FOR_BROWSER_SKIP = 200
MIN_SOURCES_TO_COMPLETE = 3
MIN_SOURCES_TO_COMPLETE_WITH_GOOD = 2

# ── Limits & sizes ─────────────────────────────────────────────
TRACE_DETAIL_MAX_LENGTH = 2500
MAX_CONTENT_PER_SOURCE_STORED = 30_000
KNOWLEDGE_SUMMARY_MAX_CHARS = 2000
KNOWLEDGE_SUMMARY_PER_SOURCE_CHARS = 200
OBSERVATION_SNIPPET_LENGTH = 300
MAX_QUERY_WORD_COUNT = 8
FETCH_RETRIES = 2
BROWSER_SETTLE_DELAY = 0.6

# ── Junk domains to skip (search engines, redirect pages, etc.) ─
JUNK_DOMAINS: frozenset[str] = frozenset({
    "google.com", "google.be", "google.nl", "google.co.uk", "google.de",
    "google.fr", "google.co.in", "google.ca", "google.com.au",
    "bing.com", "yahoo.com", "search.yahoo.com", "yandex.com", "yandex.ru",
    "baidu.com", "duckduckgo.com", "search.brave.com", "ecosia.org",
    "startpage.com", "ask.com", "aol.com",
})
