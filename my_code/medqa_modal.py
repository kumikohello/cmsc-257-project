#!/usr/bin/env python3
"""
Modal-based MedQA evaluation with uncertainty.

This script runs a HuggingFace LLM on MedQA inside Modal with a GPU.

For each question, it:
  - Builds a prompt ending with "Final answer:"
  - Gets a deterministic prediction (temperature=0)
  - Computes log probability of the predicted answer token
  - Samples M answers at higher temperature to estimate an option-level distribution
  - Computes predictive entropy over {A,B,C,D,E}
  - Returns per-question results to the local machine, where they are saved as a CSV.

Columns in CSV:
  question_id, gold, pred_det, correct_det, logprob_det,
  n_samples, entropy, p_A, p_B, p_C, p_D, p_E

Usage example (from your terminal):

  modal run medqa_modal.py \\
    --local_data_path /Users/UChicago/classes/CMSC_25700/Project/data_clean/questions/US/test.jsonl \\
    --model_name google/gemma-2b-it \\
    --num_examples 50 \\
    --m_samples 20 \\
    --sample_temperature 0.8 \\
    --output_csv gemma_medqa_us_test.csv
"""
import os
import math
from typing import List, Dict, Any, Optional

import modal

app = modal.App("medqa-entropy-eval")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "pandas",
        "numpy",
    )
)

# Updated GPU syntax per Modal warning
GPU_CONFIG = "A10G"   # or "T4" etc, depending on your workspace


# ---------- Local helper to load MedQA JSONL ---------- #

def load_medqa_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load MedQA jsonl file into a list of dicts."""
    import json

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


# ---------------------- Remote function ---------------------- #

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("hf-token")]
)
def evaluate_medqa_remote(
    examples: List[Dict[str, Any]],
    model_name: str,
    m_samples: int,
    sample_temperature: float,
) -> List[Dict[str, Any]]:
    """
    Runs inside Modal on a GPU.

    Takes:
      - examples: list of MedQA dicts (already loaded locally)
      - model_name: HF model id
      -m_samples: number of samples for entropy
      - sample_temperature: temperature for sampling
    Returns:
      - List of per-question result dicts (rows for CSV).
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # ---------- Helper functions (inside remote so Modal sees them) ---------- #

    def build_base_prompt(example: Dict[str, Any]) -> str:
        """
        Build a multiple-choice prompt for the MedQA example.

        We end with "Final answer:" so the next token is the answer letter.
        """
        question = example["question"]
        options_dict = example["options"]  # {"A": "...", "B": "...", ...}

        option_keys = sorted(options_dict.keys())
        options_str_lines = [f"{k}. {options_dict[k]}" for k in option_keys]
        options_str = "\n".join(options_str_lines)

        prompt = f"""
You are an expert physician taking a multiple-choice medical exam.

Question:
{question}

Options:
{options_str}

First, think step by step.
Then, on the last line, output your final answer in the format:
Final answer: 
""".rstrip()
        return prompt

    def generate_deterministic_answer(
        model,
        tokenizer,
        prompt: str,
        device: torch.device,
        max_new_tokens: int = 16,
    ) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

    def parse_final_answer_letter(text: str) -> Optional[str]:
        marker = "Final answer:"
        idx = text.rfind(marker)
        if idx == -1:
            return None
        after = text[idx + len(marker):].strip()
        for ch in after:
            if ch.upper() in ["A", "B", "C", "D", "E"]:
                return ch.upper()
        return None

    def compute_logprob_of_letter(
        model,
        tokenizer,
        prompt: str,
        letter: str,
        device: torch.device,
    ) -> Optional[float]:
        """Compute log p(letter | prompt) at next-token position."""
        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # [1, seq_len, vocab_size]

        last_logits = logits[0, -1, :]
        log_probs = F.log_softmax(last_logits, dim=-1)

        # Try " space + letter" first
        cand = " " + letter
        cand_ids = tokenizer.encode(cand, add_special_tokens=False)
        token_id = None
        if len(cand_ids) == 1:
            token_id = cand_ids[0]
        else:
            cand_ids = tokenizer.encode(letter, add_special_tokens=False)
            if len(cand_ids) == 1:
                token_id = cand_ids[0]

        if token_id is None:
            return None

        logprob = float(log_probs[token_id].item())
        return logprob

    def sample_answers(
        model,
        tokenizer,
        prompt: str,
        device: torch.device,
        M: int = 20,
        temperature: float = 0.8,
        max_new_tokens: int = 16,
    ) -> list[str]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                num_return_sequences=M,
                pad_token_id=tokenizer.eos_token_id,
            )
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return texts

    def compute_entropy_from_counts(counts: Dict[str, int], letters: list[str]) -> float:
        total = sum(counts.get(l, 0) for l in letters)
        if total == 0:
            return float("nan")
        entropy = 0.0
        for l in letters:
            c = counts.get(l, 0)
            if c <= 0:
                continue
            p = c / total
            entropy -= p * math.log(p)
        return entropy

    # ---------- Load model ---------- #

    print(f"[Modal] Loading model: {model_name}")
    hf_token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"[Modal] Got {len(examples)} examples.")
    option_letters = ["A", "B", "C", "D", "E"]
    rows: List[Dict[str, Any]] = []

    for i, ex in enumerate(examples):
        base_prompt = build_base_prompt(ex)

        # 1) Deterministic answer
        det_text = generate_deterministic_answer(
            model, tokenizer, base_prompt, device=device
        )
        pred_det = parse_final_answer_letter(det_text)
        gold = ex["answer_idx"].strip().upper()

        correct_det = int(pred_det == gold) if pred_det is not None else 0

        # 2) Logprob of predicted letter
        if pred_det is not None:
            logprob_det = compute_logprob_of_letter(
                model, tokenizer, base_prompt, pred_det, device=device
            )
        else:
            logprob_det = None

        # 3) Sampling for entropy
        sampled_texts = sample_answers(
            model,
            tokenizer,
            base_prompt,
            device=device,
            M=m_samples,
            temperature=sample_temperature,
        )
        sample_letters = [parse_final_answer_letter(t) for t in sampled_texts]

        counts = {L: 0 for L in option_letters}
        for L in sample_letters:
            if L in counts:
                counts[L] += 1

        entropy = compute_entropy_from_counts(counts, option_letters)

        total_samples = sum(counts.values())
        if total_samples > 0:
            probs = {L: counts[L] / total_samples for L in option_letters}
        else:
            probs = {L: float("nan") for L in option_letters}

        row = {
            "question_id": i,
            "gold": gold,
            "pred_det": pred_det,
            "correct_det": correct_det,
            "logprob_det": logprob_det,
            "n_samples": total_samples,
            "entropy": entropy,
            "p_A": probs["A"],
            "p_B": probs["B"],
            "p_C": probs["C"],
            "p_D": probs["D"],
            "p_E": probs["E"],
        }
        rows.append(row)

        print(
            f"[Modal] Example {i+1}/{len(examples)} | "
            f"gold={gold}, pred={pred_det}, correct={correct_det}, "
            f"logprob={logprob_det}, entropy={entropy:.3f}"
        )

    return rows


# ---------------------- Local entrypoint ---------------------- #

@app.local_entrypoint()
def main(
    local_data_path: str,
    model_name: str = "google/gemma-2b-it",
    num_examples: int = 50,
    m_samples: int = 20,
    sample_temperature: float = 0.8,
    output_csv: str = "medqa_results_modal.csv",
):
    """
    This runs on your local machine:
      - Loads the MedQA jsonl from local_data_path
      - Truncates to num_examples
      - Calls the remote GPU function with the examples
      - Saves the returned rows as a CSV locally
    """
    import pandas as pd

    print(f"[Local] Loading MedQA from: {local_data_path}")
    examples = load_medqa_jsonl(local_data_path, limit=num_examples)
    print(f"[Local] Loaded {len(examples)} examples.")

    print("[Local] Calling Modal remote function...")
    rows = evaluate_medqa_remote.remote(
        examples=examples,
        model_name=model_name,
        m_samples=m_samples,
        sample_temperature=sample_temperature,
    )

    print(f"[Local] Got {len(rows)} rows back. Saving to {output_csv}...")
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print("[Local] Done. Head of dataframe:")
    print(df.head())
