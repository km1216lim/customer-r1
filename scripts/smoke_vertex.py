"""One-shot smoke test for Vertex AI Gemini 2.5 Pro via the same SDK
(google.genai) that data/synthesize_rationales.py uses. Run from repo root:

    python scripts/smoke_vertex.py

If this succeeds, Phase 2 (rationale synthesis) can be started.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

required = ["GOOGLE_APPLICATION_CREDENTIALS", "GCP_PROJECT_ID", "GCP_LOCATION"]
missing = [k for k in required if not os.environ.get(k)]
if missing:
    sys.exit(f"Missing env vars: {missing}. Check .env.")

key_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
if not Path(key_path).exists():
    sys.exit(f"Service-account key file not found: {key_path}")

print(f"[env] project={os.environ['GCP_PROJECT_ID']} location={os.environ['GCP_LOCATION']}")
print(f"[env] HTTPS_PROXY={os.environ.get('HTTPS_PROXY', '(unset)')}")

from google import genai

client = genai.Client(
    vertexai=True,
    project=os.environ["GCP_PROJECT_ID"],
    location=os.environ["GCP_LOCATION"],
)

print("[call] generate_content model=gemini-2.5-pro ...")
resp = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="In one short sentence, say hello and confirm you are Gemini 2.5 Pro.",
)
print("[ok] response:")
print((resp.text or "").strip())
