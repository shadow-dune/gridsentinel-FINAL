"""Unstructured alert text from a configured RSS/Atom feed (e.g. Meteoalarm Germany). Raw text is DATA only."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Tuple

import feedparser

from config.settings import Settings
from data.http import HttpFailure, get_json
from data.schemas import AlertFeed, AlertItem, SignalKind


def fetch_alerts(s: Settings, max_items: int = 8) -> Tuple[AlertFeed, str, str, SignalKind]:
    if not s.ALERT_FEED_URL:
        raise HttpFailure("ALERT_FEED_URL not configured")
    text = get_json(s.ALERT_FEED_URL, timeout=s.HTTP_TIMEOUT_SECONDS, retries=s.HTTP_MAX_RETRIES, as_text=True)
    parsed = feedparser.parse(text)
    if not parsed.entries:
        return AlertFeed(alerts=[]), "Alert RSS feed (0 entries)", s.ALERT_FEED_URL, SignalKind.MEASURED
    items = []
    for e in parsed.entries[:max_items]:
        summary = re.sub(r"<[^>]+>", "", getattr(e, "summary", "") or getattr(e, "title", "")).strip()[:400]
        items.append(AlertItem(title=getattr(e, "title", "").strip()[:200], summary=summary,
                               published=getattr(e, "published", datetime.now(timezone.utc).isoformat()),
                               link=getattr(e, "link", s.ALERT_FEED_URL), source_feed=s.ALERT_FEED_URL))
    return AlertFeed(alerts=items), "Alert RSS feed", s.ALERT_FEED_URL, SignalKind.MEASURED
