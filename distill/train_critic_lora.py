import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


PROMPT = """You are a post-action critic for a mobile GUI agent.

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_history(history: Any) -> str:
    if isinstance(history, list):
        return "\n".join(str(item) for item in history) if history else "(empty)"
    return str(history or "(empty)")


def build_prompt(row: dict[str, Any]) -> str:
    data = row["input"]
    return PROMPT.format(
        goal=data.get("goal", ""),
        history=format_history(data.get("history", [])),
        before_ui=data.get("before_ui", ""),
        action=data.get("action", ""),
        after_ui=data.get("after_ui", ""),
    )


def build_answer(row: dict[str, Any]) -> str:
    return json.dumps(row["output"], ensure_ascii=False, separators=(",", ":"))


class CriticDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        prompt = build_prompt(row)
        answer = build_answer(row) + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids

        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class DataCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = pad_sequence(
            [x["input_ids"] for x in features],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = pad_sequence(
            [x["attention_mask"] for x in features],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [x["labels"] for x in features],
            batch_first=True,
            padding_value=-100,
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description="QLoRA/SFT train the VC post-action critic model.")
    parser.add_argument("--model", required=True, help="Base model path or Hugging Face model id.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_config,
    )
    model.config.use_cache = False
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_rows = read_jsonl(data_dir / "critic_distill_train.jsonl")
    val_rows = read_jsonl(data_dir / "critic_distill_val.jsonl")
    train_ds = CriticDataset(train_rows, tokenizer, args.max_length)
    val_ds = CriticDataset(val_rows, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    (output_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Saved LoRA adapter to {output_dir / 'adapter'}")


if __name__ == "__main__":
    main()
