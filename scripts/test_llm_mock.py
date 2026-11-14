#!/usr/bin/env python3
"""Verify the optional LLM layer without an API key.

Starts a local OpenAI-compatible mock server, feeds it a few valid and invalid
designs, and checks that only novel, renderable ones survive validation.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DESIGNS = {
    "designs": [
        {
            "components": ["雨", "心"],
            "structure": "⿱",
            "reading": "yǔxīn",
            "gloss": "rain-heart; longing in weather",
            "rationale": "Rain above the heart turns weather into feeling.",
        },
        {
            "components": ["火", "永"],
            "structure": "⿰",
            "reading": "yǒnghuǒ",
            "gloss": "an eternal fire",
            "rationale": "Fire that lasts.",
        },
        {
            "components": ["不", "心"],
            "structure": "⿱",
            "reading": "bùxīn",
            "gloss": "invalid design",
            "rationale": "Should be rejected.",
        },
        {
            "components": ["字", "心"],
            "structure": "⿰",
            "reading": "zìxīn",
            "gloss": "a digital heart",
            "rationale": "Writing carrying feeling.",
        },
    ]
}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        content = json.dumps(DESIGNS, ensure_ascii=False)
        body = json.dumps(
            {"choices": [{"message": {"content": content}}]}, ensure_ascii=False
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    os.environ["FORGE_LLM_API_KEY"] = "mock-key"
    os.environ["FORGE_LLM_BASE_URL"] = f"http://127.0.0.1:{server.server_address[1]}"
    os.environ["FORGE_LLM_MODEL"] = "mock-model"

    from forge import llm
    from forge.engine import ForgeEngine

    cache = llm._load_cache()
    for key in list(cache):
        if key.startswith("mock-model::"):
            del cache[key]
    llm._save_cache(cache)

    try:
        engine = ForgeEngine()
        result = engine.forge("longing in the rain", count=4, use_llm=True)
        assert result.llm_used, "LLM candidates were not used"
        assert any(c.origin == "llm" for c in result.candidates), "no LLM candidate"
        assert not any(c.parts == ("不", "心") for c in result.candidates), (
            "function character was not rejected"
        )
        assert any(c.shapes == ("永", "灬") for c in result.candidates), (
            "existing 火永 did not fall back to 永灬"
        )
        for candidate in result.candidates:
            signature = (candidate.structure or "") + "".join(candidate.shapes)
            assert not signature or signature not in engine.real, (
                f"candidate already exists: {signature}"
            )

        print("LLM mock check passed. Candidates:")
        for candidate in result.candidates:
            print(
                f"  {candidate.origin:13s} {candidate.structure} "
                f"{' + '.join(candidate.shapes)}"
                + (f"  [{candidate.gloss}]" if candidate.gloss else "")
            )
        return 0
    finally:
        server.shutdown()
        cache = llm._load_cache()
        for key in list(cache):
            if key.startswith("mock-model::"):
                del cache[key]
        llm._save_cache(cache)


if __name__ == "__main__":
    sys.exit(main())
