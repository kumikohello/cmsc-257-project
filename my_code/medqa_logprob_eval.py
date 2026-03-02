#!/usr/bin/env python3
import json
import argparse
import csv
import math
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


LETTERS = ["A", "B", "C", "D", "E"]


def load_medqa_jsonl(path: str, limit: Optional[int] = None):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            example = json.loads(line.strip())
            data.append(example)
    return data


def build_prompt(example: Dict[str, Any]) -> str:
    question = example["question"]
    options_dict = example["options"]
    option_keys = sorted(options_dict.keys())

    options_str = "\n".join(
        [f"{k}. {options_dict[k]}" for k in option_keys]
    )

    prompt = f"""
You are an expert physician taking a multiple-choice medical exam.

Question:
{question}

Options:
{options_str}

Think carefully and output ONLY the letter of the correct answer.
Answer:
"""
    return prompt.strip()


def get_option_logprobs(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[:, -1, :]  # last token
    log_probs = F.log_softmax(logits, dim=-1)

    option_logprobs = {}
    for letter in LETTERS:
        token_id = tokenizer.encode(letter, add_special_tokens=False)[0]
        option_logprobs[letter] = log_probs[0, token_id].item()

    return option_logprobs


def compute_entropy(logprob_dict):
    probs = [math.exp(lp) for lp in logprob_dict.values()]
    total = sum(probs)
    probs = [p / total for p in probs]

    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    return entropy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_name", default="google/gemma-2b-it")
    parser.add_argument("--num_examples", type=int, default=50)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    examples = load_medqa_jsonl(args.data_path, args.num_examples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    rows = []

    for i, ex in enumerate(examples):
        prompt = build_prompt(ex)
        logprobs = get_option_logprobs(model, tokenizer, prompt, device)

        # Normalize
        probs = {k: math.exp(v) for k, v in logprobs.items()}
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        pred_letter = max(probs, key=probs.get)
        gold_letter = ex["answer_idx"].strip().upper()
        correct = int(pred_letter == gold_letter)

        entropy = compute_entropy(logprobs)

        row = {
            "question_id": i,
            "gold_letter": gold_letter,
            "pred_letter": pred_letter,
            "correct": correct,
            "entropy": entropy,
        }

        for letter in LETTERS:
            row[f"logprob_{letter}"] = logprobs[letter]
            row[f"prob_{letter}"] = probs[letter]

        rows.append(row)

        print(f"Example {i+1}: Pred={pred_letter}, Gold={gold_letter}, Correct={correct}, Entropy={entropy:.3f}")

    # Save CSV
    fieldnames = rows[0].keys()
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to {args.output_csv}")


if __name__ == "__main__":
    main()
