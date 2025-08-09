from __future__ import annotations

from typing import Any, Dict, List, Optional
import asyncio
import time
import json
import jsonschema
import random

from persistence import RunStore, BlobStore, RunEvent, EventType


def _last_json_obj(text: str):
    """Return the last valid JSON object embedded in ``text``, else None."""
    dec = json.JSONDecoder()
    s = text or ""
    last = None
    i = 0
    while True:
        while i < len(s) and s[i].isspace():
            i += 1
        if i >= len(s):
            break
        try:
            obj, end = dec.raw_decode(s, i)
            last = obj
            i = end
        except json.JSONDecodeError:
            i += 1
    return last


# =============== Prompt Packet & Schemas ===============

PROMPT_PACKET = """\
SYSTEM:
You are {role}. Follow the RULES strictly.

RULES:
- Think privately; do not include analysis in output.
- Return JSON that matches OUTPUT_SCHEMA exactly.
- If information is missing, set fields to null and add an item to `notes`.
- Never invent file paths or symbols; use ones provided in CONTEXT.
- If you need sub-steps, propose them in `children`, do not execute them.

TASK_INTENT: {intent}

IDENTIFIERS:
run_id: {run_id}
finding_id: {finding_id}
condition_id: {condition_id}
parent_condition_id: {parent_condition_id}

CONTEXT:
{context_block}

OUTPUT_SCHEMA (JSON Schema):
{json_schema}

OUTPUT_EXAMPLE:
{few_shot_example}

RESPONSE:
Return ONLY the JSON object. No prose, no backticks.
"""


VALIDATION_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["status", "evidence", "children", "notes"],
    "properties": {
        "status": {"type": "string", "enum": ["SATISFIED", "VIOLATED", "UNKNOWN"]},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["summary", "source", "strength"],
                "properties": {
                    "summary": {"type": "string", "minLength": 1},
                    "source": {"type": "string"},
                    "locations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["file"],
                            "properties": {
                                "file": {"type": "string"},
                                "start_line": {"type": ["integer", "null"]},
                                "end_line": {"type": ["integer", "null"]},
                                "symbol": {"type": ["string", "null"]}
                            }
                        }
                    },
                    "strength": {"type": "number", "minimum": 0, "maximum": 1},
                    "raw_refs": {"type": "object"},
                    "witness": {"type": ["string", "null"]}
                }
            }
        },
        "children": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["text", "plan_params"],
                "properties": {
                    "text": {"type": "string"},
                    "plan_params": {"type": "object"}
                }
            }
        },
        "notes": {"type": "array", "items": {"type": "string"}}
    }
}


SCOUT_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["findings", "notes"],
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["claim", "origin_file", "root_conditions"],
                "properties": {
                    "claim": {"type": "string"},
                    "origin_file": {"type": "string"},
                    "root_conditions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["text", "plan_params"],
                            "properties": {
                                "text": {"type": "string"},
                                "plan_params": {"type": "object"}
                            }
                        }
                    }
                }
            }
        },
        "notes": {"type": "array", "items": {"type": "string"}}
    }
}


LLM_FAILURES: List[str] = []


async def call_llm_with_schema(
    llm,
    *,
    system,
    messages,
    schema,
    max_retries=2,
    temperature=0,
    top_p=1,
    max_tokens=1024,
    run_store: Optional[RunStore] = None,
    blob_store: Optional[BlobStore] = None,
    run_id: str = "",
    task_id: str = "",
    finding_id: str = "",
    condition_id: str = "",
):
    t0 = time.time()
    attempt = 0
    while True:
        resp = await llm.complete(
            system=system,
            messages=messages,
            json_schema=schema,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        text = resp.get("text") if isinstance(resp, dict) else (resp or "")
        obj = _last_json_obj(text)

        if obj is None:
            messages.append(
                {
                    "role": "system",
                    "content": "Your previous reply did not contain a JSON object. Return ONLY JSON.",
                }
            )
            attempt += 1
            if run_store:
                run_store.append(
                    RunEvent(
                        ts=time.time(),
                        run_id=run_id,
                        type=EventType.LLM_RETRY,
                        data={
                            "task_id": task_id,
                            "finding_id": finding_id,
                            "condition_id": condition_id,
                            "attempt": attempt,
                            "reason": "no_json_object",
                        },
                    )
                )
            if attempt > max_retries:
                LLM_FAILURES.append("no_json_object")
                raise ValueError("LLM returned no JSON object")
            await asyncio.sleep(min(1.5 ** attempt, 8) + random.random() * 0.25)
            continue

        try:
            jsonschema.validate(obj, schema)
        except Exception as e:
            messages.append(
                {
                    "role": "system",
                    "content": f"Your JSON was invalid ({e}). Fix it to match the schema exactly.",
                }
            )
            attempt += 1
            if run_store:
                run_store.append(
                    RunEvent(
                        ts=time.time(),
                        run_id=run_id,
                        type=EventType.LLM_RETRY,
                        data={
                            "task_id": task_id,
                            "finding_id": finding_id,
                            "condition_id": condition_id,
                            "attempt": attempt,
                            "reason": "schema_validation_error",
                            "error": str(e),
                        },
                    )
                )
            if attempt > max_retries:
                LLM_FAILURES.append(str(e))
                raise
            await asyncio.sleep(min(1.5 ** attempt, 8) + random.random() * 0.25)
            continue

        latency = time.time() - t0
        prompt_bytes = json.dumps({"system": system, "messages": messages}).encode()
        response_bytes = (text or "").encode()
        prompt_sha = blob_store.put(prompt_bytes) if blob_store else None
        response_sha = blob_store.put(response_bytes) if blob_store else None
        usage = {
            "latency_s": latency,
            "prompt_sha": prompt_sha,
            "response_sha": response_sha,
            "prompt_tokens": len(prompt_bytes.split()),
            "response_tokens": len(response_bytes.split()),
        }
        return obj, usage
