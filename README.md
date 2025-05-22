# palin-bench: Reversing Words with Hugging Face Language Models

This repo fine-tunes Hugging Face causal language models to **reverse input words** — a deliberately unnatural task (character-level transformations like reversing a word are not meaningful language modeling tasks) that serves as a lightweight benchmark for evaluating performance of Apple M-Series devices for fine-tuning via MPS.

---

## 💻 Benchmarks

| Device           | Model                | Tokens/sec | Final Loss (5 Epochs) |
|------------------|----------------------|------------|------------------------|
| M2 Max (32 GB)    | Qwen/Qwen3-0.6B-Base | 2.74       | x.xx                   |

> 🧪 **PRs welcome!** Add your own results (any model / any hardware).

---

## How It Works

- **Input:** a single word (e.g., `apple`)
- **Target:** the reversed word with EOS (e.g., `elppa <eos>`)
- **Objective:** Teach the model to learn this transformation from scratch