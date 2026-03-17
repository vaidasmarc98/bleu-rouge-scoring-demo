# LLM Evaluation Dashboard

An interactive dashboard for scoring LLM responses against reference answers using BLEU and ROUGE metrics, built with Streamlit and Plotly.

---

## What Problem This Solves

Evaluating LLM output quality at scale requires standardized, reproducible metrics. BLEU and ROUGE are the industry standard for measuring how closely a model's response resembles a reference answer — widely used in summarization, translation, and QA tasks. This dashboard makes those scores visual and interactive, enabling rapid comparison across prompts and categories.

---

## Features

- **BLEU scoring** — measures n-gram precision against reference answers
- **ROUGE-1, ROUGE-2, ROUGE-L** — measures recall-oriented overlap (unigram, bigram, longest subsequence)
- **Pass/fail thresholds** — configurable per-metric cutoffs
- **Score distribution charts** — histogram and box plots per metric
- **Category breakdown** — grouped bar charts and radar charts by prompt category
- **Response inspector** — side-by-side view of prompt, reference, model response, and per-metric scores
- **Re-run button** — re-queries the model and refreshes all charts live

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| Streamlit | Interactive dashboard UI |
| sacrebleu | BLEU metric |
| rouge-score | ROUGE-1, ROUGE-2, ROUGE-L |
| Anthropic API | Model under test (Claude Haiku) |
| pandas | Data manipulation |
| Plotly | Interactive charts |

---

## Project Structure

```
llm-eval-dashboard/
├── data/
│   └── reference_sets.json      # prompts + reference answers
├── evaluator/
│   └── scorer.py                # LLM querying + metric computation
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Running Locally

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Set API keys**
```bash
export ANTHROPIC_API_KEY=your_key_here
```

**3. Launch the dashboard**
```bash
streamlit run dashboard/app.py
```

The dashboard opens automatically at `http://localhost:8501`.

---

## Configuration

Scoring thresholds and model are set at the top of `evaluator/scorer.py`:

| Variable | Default | Description |
|---|---|---|
| `MODEL` | claude-haiku-4-5-20251001 | Model under test |
| `BLEU_PASS_THRESHOLD` | 15.0 | Min BLEU score to pass (0–100) |
| `ROUGE1_PASS_THRESHOLD` | 0.35 | Min ROUGE-1 F1 to pass |
| `ROUGE2_PASS_THRESHOLD` | 0.15 | Min ROUGE-2 F1 to pass |
| `ROUGEL_PASS_THRESHOLD` | 0.30 | Min ROUGE-L F1 to pass |

---

## Adding Reference Sets

Add entries to `data/reference_sets.json`:

```json
{
  "id": "rs_011",
  "category": "your_category",
  "prompt": "Your prompt here",
  "reference": "The ideal reference answer the model should approximate."
}
```

Categories currently supported: `summarization`, `explanation`. Add any category — the dashboard groups automatically.

---

## Dashboard Screenshots

| View | Description |
|---|---|
| Summary metrics | Pass rate, avg BLEU, avg ROUGE scores at a glance |
(screenshots/Capture1.png)
| Score distributions & Category breakdown  | Histogram of BLEU scores, box plots of ROUGE scores | | Grouped bar chart + radar chart by prompt category
(screenshots/Capture2.png)
| Response inspector | Side-by-side prompt / reference / model output with per-metric pass/fail |
(screenshots/Capture3.png)
---

## Author

Vaidas Marcinkevicius · [linkedin.com/in/vaidasmarc](https://linkedin.com/in/vaidasmarc)
