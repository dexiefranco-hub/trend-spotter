"""
Trend Spotter — Agent 1
Runs weekly via GitHub Actions. Uses Claude + web search to gather:
  1. Amazon rising products (Movers & Shakers / Best Sellers signals)
  2. Google Trends rising searches
  3. TikTok trend signals (breakout hashtags + product buzz)
Writes everything to data.json, which the dashboard (index.html) displays.
"""

import json
import os
import re
import sys
from datetime import datetime, timezone

from anthropic import Anthropic

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

PROMPT = """You are a product trend research agent for a dropshipping business.
Today's date: {today}. You are reporting on the week of {week_start} to {today}.

STRICT RECENCY RULE: Only include items supported by information from the LAST 7 DAYS.
Include the current year in your search queries. If a search result is older than
about two weeks, do not use it. Prefer "this week" and "right now" signals.

Use web search to research CURRENT data for three separate panels:

PANEL 1 — AMAZON: Products currently rising on Amazon (Movers & Shakers style
signals, best seller climbers). Focus on consumer products a dropshipper could
sell (gadgets, home, beauty, fitness, pets). Find 6-8 items.

PANEL 2 — GOOGLE TRENDS: Product search terms rising in the US right now —
strictly things people want to BUY (e.g. "portable neck fan", "walking pad").
Exclude cultural topics, lifestyle movements, events, or celebrity names;
those are handled by a separate agent. Find 6-8 rising product searches.

PANEL 3 — TIKTOK: Specific PRODUCTS getting strong buying traction on TikTok
right now — items being tagged, demoed, and sold in videos (TikTok Shop style
signals). Strictly products, not hashtags or cultural trends. Find 6-8 items.

Then PANEL 4 — CROSS-PLATFORM WINNERS: For your top Amazon products, actively
search whether the SAME product is also selling on TikTok Shop (e.g. search
"<product> TikTok Shop"). Products confirmed moving on BOTH platforms go in
"aligned" with sources ["amazon","tiktok"]. These are the strongest buy
signals because demand is validated on two platforms. Also include any
Google Trends overlaps. Aim for 3-5 aligned items.

Respond with ONLY valid JSON, no markdown fences, no commentary, exactly this shape:
{{
  "amazon": [{{"name": "...", "category": "...", "signal": "max 12 words", "metric": "the number, e.g. '10K+ bought past month'", "momentum": 1-100}}],
  "google_trends": [{{"name": "...", "category": "...", "signal": "max 12 words", "metric": "the number, e.g. '150K searches/mo, +40%'", "momentum": 1-100}}],
  "tiktok": [{{"name": "...", "category": "...", "signal": "max 12 words", "metric": "the number, e.g. '$25M+ GMV' or '340M views'", "momentum": 1-100}}],
  "aligned": [{{"name": "...", "sources": ["amazon","tiktok"], "why": "max 15 words", "metric": "best number across platforms"}}],
  "summary": "2-3 sentence plain-language summary of the week's biggest opportunity"
}}

METRIC RULE — CRITICAL: Every real figure you find (units bought, GMV, views,
search volume, growth %, bestseller rank) goes in the "metric" field, NOT
buried inside signal sentences. Never invent numbers. Use "" only if you
truly found no figure. Keep "signal" and "why" SHORT — they display on a
small dashboard.

momentum = your 1-100 estimate of how fast it's rising (100 = explosive).
Every item must reflect real, current information found via search."""


def extract_json(text: str):
    """Pull the JSON object out of the model's reply, tolerating fences/preamble."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model response")
    return json.loads(cleaned[start : end + 1])


def main():
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    today = now.strftime("%B %d, %Y")
    week_start = (now - timedelta(days=7)).strftime("%B %d, %Y")
    print(f"Running Trend Spotter for week {week_start} - {today}...")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 12}],
        messages=[{"role": "user", "content": PROMPT.format(today=today, week_start=week_start)}],
    )

    # Combine all text blocks (web search responses interleave tool blocks)
    full_text = "".join(
        block.text for block in response.content if block.type == "text"
    )

    try:
        data = extract_json(full_text)
    except Exception as e:
        print(f"Failed to parse agent output: {e}", file=sys.stderr)
        print("Raw output was:", full_text[:2000], file=sys.stderr)
        sys.exit(1)

    # Basic sanity checks so a bad run never wipes a good dashboard
    for key in ("amazon", "google_trends", "tiktok"):
        if not isinstance(data.get(key), list) or len(data[key]) == 0:
            print(f"Panel '{key}' came back empty — keeping previous data.", file=sys.stderr)
            sys.exit(1)

    data["updated"] = now.isoformat()
    data["updated_display"] = today
    data["week_range"] = f"{week_start} – {today}"

    with open("data.json", "w") as f:
        json.dump(data, f, indent=2)

    print("data.json written successfully.")
    print(f"Amazon: {len(data['amazon'])} | Trends: {len(data['google_trends'])} | TikTok: {len(data['tiktok'])} | Aligned: {len(data.get('aligned', []))}")


if __name__ == "__main__":
    main()
