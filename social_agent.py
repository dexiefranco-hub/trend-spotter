"""
Social Trend Radar — Agent 2
Runs weekly via GitHub Actions, 30 min after Agent 1 (Trend Spotter).
Watches CULTURAL waves forming on social media — the 2-6 week early-warning
layer that runs ahead of product sales data. For each trend it suggests
product opportunities, which feed Agent 1's product research.
Writes everything to social.json, displayed by trends.html.
"""

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

from anthropic import Anthropic

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

PROMPT = """You are a social media trend analyst for an e-commerce entrepreneur.
Today's date: {today}. You are reporting on the week of {week_start} to {today}.

STRICT RECENCY RULE: Only include trends supported by information from the LAST
7-14 DAYS. Include the current year in search queries. Skip anything stale.

Your job is the CULTURAL layer, not products: lifestyle movements, viral
formats, hobby waves, aesthetic shifts, communities suddenly growing — things
like when pickleball, cold plunges, or cozy gaming took off. These run 2-6
weeks AHEAD of product sales, so catching them early is the entire value.

Use web search to find 8-12 rising social/cultural trends across TikTok,
Instagram, Reddit, and YouTube. For each, classify its stage:
- "emerging" = tiny but accelerating, most people haven't heard of it yet
- "rising"   = clearly growing fast, early adopters all over it
- "peaking"  = mainstream now, window closing for early advantage

Respond with ONLY valid JSON, no markdown fences, no commentary, exactly this shape:
{{
  "trends": [
    {{
      "name": "...",
      "category": "fitness / home / food / fashion / hobby / tech / lifestyle / other",
      "stage": "emerging | rising | peaking",
      "signal": "what's happening and where, one or two sentences",
      "metric": "real figure found in search, e.g. '480M hashtag views, up 3x this month' — never invent; use \\"\\" if none found",
      "momentum": 1-100,
      "product_ideas": ["2-4 concrete dropshippable product opportunities this trend creates"]
    }}
  ],
  "watchlist_note": "1-2 sentences: which ONE trend deserves the closest watch next week and why",
  "summary": "2-3 sentence plain-language summary of the week's cultural picture"
}}

METRIC RULE: metrics must be REAL figures found via search (hashtag views,
subscriber growth, search growth %). Never invent numbers.
Prioritize EMERGING and RISING trends — those are the money. Include at most
2-3 peaking trends for context."""


def extract_json(text: str):
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model response")
    return json.loads(cleaned[start : end + 1])


def main():
    now = datetime.now(timezone.utc)
    today = now.strftime("%B %d, %Y")
    week_start = (now - timedelta(days=7)).strftime("%B %d, %Y")
    print(f"Running Social Trend Radar for week {week_start} - {today}...")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=5000,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}],
        messages=[{"role": "user", "content": PROMPT.format(today=today, week_start=week_start)}],
    )

    full_text = "".join(
        block.text for block in response.content if block.type == "text"
    )

    try:
        data = extract_json(full_text)
    except Exception as e:
        print(f"Failed to parse agent output: {e}", file=sys.stderr)
        print("Raw output was:", full_text[:2000], file=sys.stderr)
        sys.exit(1)

    if not isinstance(data.get("trends"), list) or len(data["trends"]) == 0:
        print("Trends came back empty — keeping previous data.", file=sys.stderr)
        sys.exit(1)

    data["updated"] = now.isoformat()
    data["updated_display"] = today
    data["week_range"] = f"{week_start} – {today}"

    with open("social.json", "w") as f:
        json.dump(data, f, indent=2)

    stages = [t.get("stage", "?") for t in data["trends"]]
    print(f"social.json written: {len(data['trends'])} trends "
          f"(emerging: {stages.count('emerging')}, rising: {stages.count('rising')}, peaking: {stages.count('peaking')})")


if __name__ == "__main__":
    main()
