"""Minimal Vertex AI Gemini API smoke test for local Windows PCs.

Companion to docs/gemini_api_local_setup.md. Designed to verify that:
  1. The Samsung corporate proxy is configured correctly,
  2. The service account JSON is reachable,
  3. The requested Gemini model is available in the chosen region.

Usage (after env vars are set per the setup guide):

    python scripts/gemini_text_api_test.py
    python scripts/gemini_text_api_test.py --model gemini-2.5-pro
    python scripts/gemini_text_api_test.py --model gemini-3.5-flash --location global
    python scripts/gemini_text_api_test.py --prompt "오늘 날씨를 시 형식으로 써줘"
    python scripts/gemini_text_api_test.py --credentials C:\path\to\key.json

All CLI flags are optional — the script falls back to GOOGLE_APPLICATION_CREDENTIALS
from the environment and uses gemini-2.5-flash / us-central1 by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--credentials",
        help="Path to the GCP service account JSON. If omitted, read from the "
             "GOOGLE_APPLICATION_CREDENTIALS environment variable.",
    )
    ap.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Vertex AI Gemini model ID. Defaults to gemini-2.5-flash (stable, cheap). "
             "Common alternatives: gemini-2.5-pro, gemini-3.5-flash (needs --location global).",
    )
    ap.add_argument(
        "--location",
        default="us-central1",
        help="Vertex AI region. Defaults to us-central1. Use 'global' for newer "
             "models like gemini-3.5-flash that are not deployed to regional endpoints.",
    )
    ap.add_argument(
        "--project",
        help="GCP project ID. If omitted, read from the service account JSON's "
             "project_id field (recommended — keeps the script generic).",
    )
    ap.add_argument(
        "--prompt",
        default="한 문장으로 자기소개를 해줘.",
        help="The prompt to send to the model.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("gemini_text_output.json"),
        help="Where to save the JSON record of (timestamp, model, prompt, response). "
             "Pass an empty string to skip saving.",
    )
    args = ap.parse_args()

    # --- 1. credentials -----------------------------------------------------
    if args.credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.credentials
    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path:
        sys.exit(
            "[gemini-test] GOOGLE_APPLICATION_CREDENTIALS not set.\n"
            "  Either pass --credentials <path-to-json> or set the env var:\n"
            "    $env:GOOGLE_APPLICATION_CREDENTIALS = 'C:\\path\\to\\key.json'"
        )
    if not Path(cred_path).exists():
        sys.exit(f"[gemini-test] credentials file not found: {cred_path}")

    # --- 2. resolve GCP project (from JSON if not passed) ------------------
    project = args.project
    if not project:
        try:
            with open(cred_path, "r", encoding="utf-8") as f:
                project = json.load(f).get("project_id")
        except Exception as e:  # noqa: BLE001 — any read failure stops us cleanly
            sys.exit(f"[gemini-test] could not read project_id from {cred_path}: {e}")
        if not project:
            sys.exit(
                "[gemini-test] service account JSON has no project_id field. "
                "Pass --project <PROJECT_ID> explicitly."
            )

    # --- 3. proxy sanity check (Samsung corp env) --------------------------
    # Not fatal if missing — the user might be off the corporate network.
    # Just print the resolved values so a misconfigured proxy is easy to spot
    # in the output.
    http_proxy = os.environ.get("HTTP_PROXY") or "(not set)"
    https_proxy = os.environ.get("HTTPS_PROXY") or "(not set)"

    # --- 4. summarize what we're about to do -------------------------------
    print("=" * 60)
    print("Vertex AI Gemini smoke test")
    print("=" * 60)
    print(f"  credentials : {cred_path}")
    print(f"  project     : {project}")
    print(f"  location    : {args.location}")
    print(f"  model       : {args.model}")
    print(f"  prompt      : {args.prompt!r}")
    print(f"  HTTP_PROXY  : {http_proxy}")
    print(f"  HTTPS_PROXY : {https_proxy}")
    print("=" * 60)

    # --- 5. import + call --------------------------------------------------
    # vertexai is imported here (not at module top) so --help works without
    # the SDK installed.
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
    except ImportError as e:
        sys.exit(
            f"[gemini-test] vertexai SDK not installed: {e}\n"
            "  Install with: pip install google-cloud-aiplatform"
        )

    print("Vertex AI 초기화 중...")
    vertexai.init(project=project, location=args.location)

    print(f"{args.model} 모델 로드 중...")
    model = GenerativeModel(args.model)

    print("텍스트 생성 요청을 보냅니다...")
    try:
        response = model.generate_content(args.prompt)
    except Exception as e:  # noqa: BLE001 — surface the Vertex error verbatim
        sys.exit(
            f"\n[gemini-test] API 호출 실패: {type(e).__name__}: {e}\n"
            "  docs/gemini_api_local_setup.md §5 (에러별 해결법) 참조."
        )

    print("\n성공! 응답:")
    print(response.text)

    # --- 6. save result ----------------------------------------------------
    if str(args.output):
        record = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "location": args.location,
            "project": project,
            "prompt": args.prompt,
            "response": response.text,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n결과를 '{args.output}'에 저장했습니다.")


if __name__ == "__main__":
    main()
