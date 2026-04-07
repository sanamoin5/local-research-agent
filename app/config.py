# ════════════════════════════════════════════════════════════════
#  Central configuration for the Local Research Agent.
#
#  Two tiers:
#    1. This file — all defaults, imported by services/reporting/pipeline.
#    2. Settings model (schemas.py, persisted in DB) — user-overridable
#       subset. When a Setting exists, it takes precedence.
#
#  ARCHITECTURE INVARIANT:
#    ALL LLM calls go through ollama_generate() and are STRICTLY
#    SEQUENTIAL — never parallel. The entire system shares a single
#    local Ollama instance. Parallel inference would overload the
#    GPU, cause OOM, and hang the machine. Every agent awaits the
#    previous one before calling the LLM.
#
#  TIMEOUT PHILOSOPHY:
#    This runs on the user's local machine. We never rush. Timeouts
#    are generous to let long inferences complete. The user can walk
#    away — the agent will keep working.
# ════════════════════════════════════════════════════════════════


# ── LLM ────────────────────────────────────────────────────────
# All timeouts are generous — this runs locally, users can wait.
# IMPORTANT: All LLM calls are SEQUENTIAL (never parallel) to avoid
# overloading the local Ollama instance. Only one inference at a time.
OLLAMA_DEFAULT_TIMEOUT = 600
DEFAULT_MAX_TOKENS = 4096
REASONING_TEMPERATURE = 0.4
SYNTHESIS_TEMPERATURE = 0.4
QUERY_GEN_TEMPERATURE = 0.5
GAP_QUERY_TEMPERATURE = 0.6
COORDINATOR_TEMPERATURE = 0.3
ANALYST_TEMPERATURE = 0.3
GOAL_TEMPERATURE = 0.3
OLLAMA_RETRIES = 2

# ── Planning (multi-agent) ──────────────────────────────────────
# 4 sequential agents: intent → decompose → expand → critique
PLANNING_TEMPERATURE = 0.4
PLANNING_TIMEOUT = 600
PLANNING_MAX_TOKENS = 800

# ── Search ─────────────────────────────────────────────────────
SEARCH_RETRY_COUNT = 5
SEARCH_BACKOFF_BASE = 4.0
SEARCH_BACKOFF_MAX = 30.0
INTER_QUERY_DELAY = 4.0
INTER_ITERATION_COOLDOWN = 5.0

# ── Synthesis ──────────────────────────────────────────────────
TOP_SOURCES_CAP = 9
ANALYST_BATCH_SIZE = 3
SOURCE_CONTENT_LENGTH = 6000   # chars fed to each per-source fact-extractor
SYNTHESIS_TIMEOUT = 1800       # bookend writer (exec summary + recs + conclusion)
SECTION_WRITER_TIMEOUT = 600   # each per-section writer gets its own call
ANALYSIS_TIMEOUT = 1200        # deep analysis/judgment — no rush
ANALYST_TIMEOUT = 600          # per-source extractor
COORDINATOR_TIMEOUT = 600      # coordinator decision
SYNTHESIS_MAX_TOKENS = 8192
SECTION_WRITER_MAX_TOKENS = 4096  # per-section writer
ANALYST_MAX_TOKENS = 2048      # per-source extraction needs less output than full analysis
DIRECT_SOURCE_THRESHOLD = 8

# ── Quality scoring ────────────────────────────────────────────
QUALITY_SCORE_GOOD = 5
QUALITY_SCORE_MEDIUM = 3
MIN_TEXT_LENGTH_FOR_BROWSER_SKIP = 200
MIN_SOURCES_TO_COMPLETE = 3
MIN_SOURCES_TO_COMPLETE_WITH_GOOD = 2
MAX_CONSECUTIVE_EMPTY_ROUNDS = 3

# ── Limits & sizes ─────────────────────────────────────────────
TRACE_DETAIL_MAX_LENGTH = 2500
MAX_CONTENT_PER_SOURCE_STORED = 30_000
KNOWLEDGE_SUMMARY_MAX_CHARS = 5000   # reasoning agent sees more of what's been found
KNOWLEDGE_SUMMARY_PER_SOURCE_CHARS = 400
OBSERVATION_SNIPPET_LENGTH = 400
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
