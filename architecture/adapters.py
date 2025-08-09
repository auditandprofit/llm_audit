from __future__ import annotations

from typing import Any, Dict, List, Optional
import json

from .models import CodeSpan


# =============== Adapters (IO / Tools) ===============

class CodebaseAdapter:
    """Minimal stub used for simulation."""

    def __init__(self, root: str):
        self.root = root

    async def read_file(self, path: str) -> str:
        return ""

    async def search(self, pattern: str, *, limit: int = 100) -> List[CodeSpan]:
        return []

    async def imports_of(self, path: str, *, limit: int = 50) -> List[str]:
        return []

    async def call_sites(self, symbol: str, *, limit: int = 50) -> List[CodeSpan]:
        return []


class WebSearchAdapter:
    async def search(self, query: str, *, limit: int = 5) -> List[Dict[str, Any]]:
        return []


# =============== LLM Stub ===============

class LLMClient:
    async def complete(
        self,
        *,
        system: str,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0,
        top_p: float = 1,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        if json_schema and "findings" in json_schema.get("properties", {}):
            text = json.dumps(
                {
                    "findings": [
                        {
                            "claim": "Simulated claim",
                            "origin_file": "start.py",
                            "root_conditions": [
                                {
                                    "text": "input is sanitized",
                                    "plan_params": {},
                                },
                                {
                                    "text": "path reaches sink",
                                    "plan_params": {"sink": "util.eval"},
                                },
                            ],
                        }
                    ],
                    "notes": [],
                }
            )
            return {"text": text}
        if json_schema and "status" in json_schema.get("properties", {}):
            text = json.dumps(
                {
                    "status": "UNKNOWN",
                    "evidence": [],
                    "children": [],
                    "notes": ["demo note"],
                }
            )
            return {"text": text}
        return {"text": "{}"}
