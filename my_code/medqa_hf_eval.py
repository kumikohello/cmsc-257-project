#!/usr/bin/env python3
import json
import argparse
from typing import List, Dict, Any, Optional
import csv

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_medqa_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load MedQA jsonl file into a list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            data.append(example)
    return data


def build_mc_prompt(example: Dict[str, Any]) -> str:
    """
    Build a multiple-choice prompt for the MedQA example.

    Show question + options in order A, B, C, D, E (or however many exist),
    and ask the model to output 'Final answer: <LETTER>' on last line.
    """
    question = example["question"]
    options_dict = example["options"]  # e.g. {"A": "...", "B": "...", ...}

    # Sort option keys to ensure consistent order: A, B, C, ...
    option_keys = sorted(options_dict.keys())

    options_str_lines = []
    for key in option_keys:
        options_str_lines.append(f"{key}. {options_dict[key]}")
    options_str = "\n".join(options_str_lines)

    prompt = f"""
You are an expert physician taking a multiple-choice medical exam.

Question:
{question}

Options:
{options_str}

First, think step by step.
Then, on the last line, output your final answer in the format:
Final answer: <LETTER>
"""
    return prompt.strip()


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    temperature: float = 0.0,
    max_new_tokens: int = 128,
) -> str:
    """Generate a response from the model given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    do_sample = temperature > 0.0

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


def parse_final_answer_letter(text: str) -> Optional[str]:
    """
    Parse the final answer letter from the model's full text output.

    We look for the LAST occurrence of 'Final answer:' and then
    extract a single capital letter A-E after it.
    """
    marker = "Final answer:"
    idx = text.rfind(marker)
    if idx == -1:
        return None

    after = text[idx + len(marker):].strip()

    for ch in after:
        if ch.upper() in ["A", "B", "C", "D", "E"]:
            return ch.upper()

    return None


def evaluate_model_on_medqa(
    data_path: str,
    model_name: str,
    num_examples: int,
    temperature: float = 0.0,
    output_csv: Optional[str] = None,
):
    # Load data
    examples = load_medqa_jsonl(data_path, limit=num_examples)
    print(f"Loaded {len(examples)} examples from {data_path}")

    # Setup model + tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct = 0
    total = 0
    results_rows = []

    for i, ex in enumerate(examples):
        prompt = build_mc_prompt(ex)
        output_text = generate_answer(
            model,
            tokenizer,
            prompt,
            device=device,
            temperature=temperature,
            max_new_tokens=128,
        )

        pred_letter = parse_final_answer_letter(output_text)
        gold_letter = ex["answer_idx"].strip().upper()  # e.g. 'C'
        meta_info = ex.get("meta_info", "")

        total += 1
        is_correct = (pred_letter == gold_letter)
        if is_correct:
            correct += 1

        print("=" * 80)
        print(f"Example {i+1}")
        print(f"Question: {ex['question'][:120]}...")
        print(f"Gold answer: {gold_letter}")
        print(f"Model output:\n{output_text}")
        print(f"Parsed prediction: {pred_letter}")
        print(f"Correct? {is_correct}")

        results_rows.append({
            "question_id": i,
            "meta_info": meta_info,
            "gold_letter": gold_letter,
            "pred_letter": pred_letter if pred_letter is not None else "",
            "correct": int(is_correct),
            "model_name": model_name,
        })

    accuracy = correct / total if total > 0 else 0.0
    print("=" * 80)
    print(f"Total examples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.3f}")

    if output_csv is not None:
        print(f"Saving per-question results to: {output_csv}")
        fieldnames = ["question_id", "meta_info", "gold_letter", "pred_letter", "correct", "model_name"]
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results_rows:
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to MedQA jsonl file (e.g. US_test_4_options.jsonl)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2b-it",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=20,
        help="Number of examples to evaluate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = deterministic).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to save per-question results as CSV (e.g. results_gemma.csv).",
    )
    args = parser.parse_args()

    evaluate_model_on_medqa(
        data_path=args.data_path,
        model_name=args.model_name,
        num_examples=args.num_examples,
        temperature=args.temperature,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
