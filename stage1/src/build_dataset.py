#!/usr/bin/env python3
"""
Stage 1 - Dataset Builder (LLM-assisted)

Reads:
  - data/prompts_base.jsonl

Writes:
  - data/prompts_instantiated.jsonl

Uses an LLM once to generate R0–R3 prompt variants per base task.
The output is frozen and reused for all later experiments.
"""
print("BUILD_DATASET IS RUNNING")

import json
from pathlib import Path
from typing import Dict, Any, List

# ---- CONFIG ----
MODEL_NAME = "gpt-4o-mini"
OUTPUT_PATH = Path("data/prompts_instantiated.jsonl")
BASE_PATH = Path("data/prompts_base.jsonl")


from openai import OpenAI
client = OpenAI()  # API key picked up from env


SYSTEM_PROMPT = """
You are generating prompt templates for a controlled MoE routing experiment.

Given a BASE TASK, produce FOUR prompt variants R0–R3.

Hard rules:
- Do NOT solve the task.
- Do NOT change any numbers or entities.
- Do NOT paraphrase the BASE TASK content; copy it verbatim inside each prompt.
- Output STRICT JSON only with keys: R0_prompt, R1_prompt, R2_prompt, R3_prompt.

Regime definitions (must follow exactly):

R0 (Zero-shot direct):
- Ask for ONLY the final answer.
- No reasoning requested.

R1 (Zero-shot chain-of-thought):
- Append a minimal reasoning cue.
- Only suggest the model to "think step by step".
- Do NOT provide a numbered algorithm or procedural checklist.

R2 (Few-shot CoT + persona):
- Start with a persona: "You are a logical auditor."
- Include exactly 3 few-shot examples in this fixed format:
  Example i:
  Q: ...
  A: Reasoning: ...
     Final answer: ...
- The examples must match the SAME STRUCTURE as the base task class (state updates / dependencies), but must use DIFFERENT numbers.
- Then include:
  Q: <BASE TASK verbatim>
  A: Reasoning: ... Final answer: ...

R3 (Structured self-consistency):
- Request exactly {M} independent attempts.
- Enforce this format:
  Attempt 1:
  Reasoning: ...
  Final answer: ...
  Attempt 2: ...
- Instruct: attempts must be independent; do not reference previous attempts.

"""

USER_PROMPT_TEMPLATE = """
Base task:
"{TASK}"

Return JSON with exactly these keys:
- R0_prompt
- R1_prompt
- R2_prompt
- R3_prompt

Each value must be a full prompt string.
"""

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def generate_regimes(base_text: str) -> Dict[str, str]:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.replace("{TASK}", base_text),
            },
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

def main() -> None:
    base_prompts = read_jsonl(BASE_PATH)
    print(f"Read {len(base_prompts)} base prompts from {BASE_PATH}")
    instantiated = []

    for base in base_prompts:
        print(f"Generating regimes for {base['id']}...")
        regimes = generate_regimes(base["base_text"])

        for regime in ["R0", "R1", "R2", "R3"]:
            instantiated.append(
                {
                    "id": f"{base['id']}_{regime}",
                    "base_id": base["id"],
                    "regime": regime,
                    "task": base.get("task"),
                    "answer_type": base.get("answer_type"),
                    "prompt_text": regimes[f"{regime}_prompt"],
                    "variant_id": 0
                }
            )

    write_jsonl(OUTPUT_PATH, instantiated)
    print(f"Wrote {len(instantiated)} prompts to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
