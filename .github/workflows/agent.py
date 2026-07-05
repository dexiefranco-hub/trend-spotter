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
Today's date: {today}.

Use web search to research CURRENT data (this week) for three separate panels:

PANEL 1 — AMAZON: Products currently rising on Amazon (Movers & Shakers style
signals, best seller climbers). Focus on consumer products a dropshipper could
sell (gadgets, home, beauty, fitness, pets). Find 6-8 items.

PANEL 2 — GOOGLE TRENDS: Search terms and product-related topics rising in the
US right now. Find 6-8 rising terms relevant to consumer products.

PANEL 3 — TIKTOK: Trending/breakout hashtags and products getting strong video
traction on TikTok right now (TikTok Creative Center style signals, viral
product buzz). Find 6-8 items.

Then create PANEL 4 — ALIGNED SIGNALS: any product or trend that appears in
TWO OR MORE of the panels above. These are the strongest buy signals.

Respond with ONLY valid JSON, no markdown fences, no commentary, exactly this shape:
{{
  "amazon": [{{"name": "...", "category": "...", "signal": "why it's rising, one sentence", "momentum": 1-100}}],
  "google_trends": [{{"name": "...", "category": "...", "signal": "one sentence", "momentum": 1-100}}],
  "tiktok": [{{"name": "...", "category": "...", "signal": "one sentence", "momentum": 1-100}}],
  "aligned": [{{"name": "...", "sources": ["amazon","tiktok"], "why": "one sentence on why this is a strong signal"}}],
  "summary": "2-3 sentence plain-language summary of the week's biggest opportunity"
}}

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
    today = datetime.now(timezone.utc).strftime("%B %d, %Y")
    print(f"Running Trend Spotter for {today}...")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}],
        messages=[{"role": "user", "content": PROMPT.format(today=today)}],
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

    data["updated"] = datetime.now(timezone.utc).isoformat()
    data["updated_display"] = today

    with open("data.json", "w") as f:
        json.dump(data, f, indent=2)

    print("data.json written successfully.")
    print(f"Amazon: {len(data['amazon'])} | Trends: {len(data['google_trends'])} | TikTok: {len(data['tiktok'])} | Aligned: {len(data.get('aligned', []))}")


if __name__ == "__main__":
    main()
