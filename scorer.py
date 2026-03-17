"""
scorer.py
Queries Claude for each reference prompt, then scores responses
using BLEU and ROUGE metrics. Returns structured results.
"""

import json
import os
from pathlib import Path

from anthropic import Anthropic
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

# ── config ────────────────────────────────────────────────────────────────────

MODEL = "claude-haiku-4-5-20251001"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "reference_sets.json"

BLEU_PASS_THRESHOLD = 15.0    # BLEU score 0–100
ROUGE1_PASS_THRESHOLD = 0.35
ROUGE2_PASS_THRESHOLD = 0.15
ROUGEL_PASS_THRESHOLD = 0.30

# ── client ────────────────────────────────────────────────────────────────────

client = Anthropic()

# ── metric instances ──────────────────────────────────────────────────────────

bleu = BLEU(effective_order=True)
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_reference_sets() -> list[dict]:
    with open(DATA_PATH) as f:
        return json.load(f)


def call_llm(prompt: str) -> str:
    message = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def compute_bleu(hypothesis: str, reference: str) -> float:
    """Return corpus BLEU score (0–100)."""
    result = bleu.sentence_score(hypothesis, [reference])
    return round(result.score, 2)


def compute_rouge(hypothesis: str, reference: str) -> dict:
    """Return ROUGE-1, ROUGE-2, ROUGE-L F1 scores (0–1)."""
    scores = rouge.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def score_sample(sample: dict, index: int, total: int) -> dict:
    """Run full scoring on a single reference sample."""
    print(f"[{index + 1}/{total}] {sample['prompt'][:60]}...")

    response = call_llm(sample["prompt"])
    bleu_score = compute_bleu(response, sample["reference"])
    rouge_scores = compute_rouge(response, sample["reference"])

    passed = (
        bleu_score >= BLEU_PASS_THRESHOLD
        and rouge_scores["rouge1"] >= ROUGE1_PASS_THRESHOLD
        and rouge_scores["rouge2"] >= ROUGE2_PASS_THRESHOLD
        and rouge_scores["rougeL"] >= ROUGEL_PASS_THRESHOLD
    )

    result = {
        "id": sample["id"],
        "category": sample["category"],
        "prompt": sample["prompt"],
        "reference": sample["reference"],
        "model_response": response,
        "bleu": bleu_score,
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "passed": passed,
    }

    status = "✓ PASS" if passed else "✗ FAIL"
    print(
        f"  {status} — BLEU: {bleu_score:.1f} | "
        f"R1: {rouge_scores['rouge1']:.3f} | "
        f"R2: {rouge_scores['rouge2']:.3f} | "
        f"RL: {rouge_scores['rougeL']:.3f}"
    )

    return result


def run_scoring() -> list[dict]:
    """Score all reference samples and return results list."""
    samples = load_reference_sets()
    results = []

    print(f"\nScoring {len(samples)} samples against reference answers...\n")

    for i, sample in enumerate(samples):
        result = score_sample(sample, i, len(samples))
        results.append(result)

    print(f"\nDone. {sum(r['passed'] for r in results)}/{len(results)} passed.\n")
    return results


if __name__ == "__main__":
    results = run_scoring()
    for r in results:
        print(json.dumps(r, indent=2))
