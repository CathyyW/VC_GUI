"""Post-action critic model for the VC closed loop."""

from __future__ import annotations

import json
import re
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


CRITIC_PROMPT = """You are a post-action critic for a mobile GUI agent.

Given goal, history, before_ui, action, and after_ui, evaluate the effect of the executed action.

Return JSON only with four fields:
- outcome: progress | failure | done | no_effect
- failure_type: none | wrong_page | wrong_target | text_error | popup_blocking | premature_complete | no_effect
- suggested_recovery: continue | navigate_back | clear_text | close_dialog | retry | complete_task
- summary_to_history: one concise sentence starting with [Critic] for the next verifier step

goal:
{goal}

history:
{history}

before_ui:
{before_ui}

action:
{action}

after_ui:
{after_ui}

JSON:
"""


def format_history(history: list[str] | str | None) -> str:
    if isinstance(history, list):
        return "\n".join(str(item) for item in history) if history else "(empty)"
    return str(history or "(empty)")


def extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def normalize_critic_output(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not raw:
        return {
            "outcome": "no_effect",
            "failure_type": "no_effect",
            "suggested_recovery": "continue",
            "summary_to_history": "[Critic] The previous action had an unclear effect; continue carefully.",
        }

    summary = str(raw.get("summary_to_history", "")).strip()
    if summary and not summary.startswith("[Critic]"):
        summary = f"[Critic] {summary}"

    return {
        "outcome": raw.get("outcome", "no_effect"),
        "failure_type": raw.get("failure_type", "none"),
        "suggested_recovery": raw.get("suggested_recovery", "continue"),
        "summary_to_history": summary,
    }


def build_critic_prompt(
    goal: str,
    history: list[str] | str | None,
    before_ui: str,
    action: str,
    after_ui: str,
) -> str:
    return CRITIC_PROMPT.format(
        goal=goal or "",
        history=format_history(history),
        before_ui=before_ui or "",
        action=action or "",
        after_ui=after_ui or "",
    )


class CriticModel:
    """Loads a distilled critic LoRA and runs post-action evaluation."""

    def __init__(
        self,
        base_model: str,
        adapter_dir: str,
        max_new_tokens: int = 160,
        use_4bit: bool = True,
    ):
        tokenizer_path = adapter_dir or base_model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = None
        if use_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quant_config,
        )
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def evaluate(
        self,
        goal: str,
        history: list[str] | str | None,
        before_ui: str,
        action: str,
        after_ui: str,
    ) -> dict[str, Any]:
        prompt = build_critic_prompt(goal, history, before_ui, action, after_ui)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        return normalize_critic_output(extract_json(text))
