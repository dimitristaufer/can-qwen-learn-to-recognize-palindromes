# palin-bench: Can M-series Macs be used to fine-tune SOTA LLMs?

This repo fine-tunes Hugging Face causal language models to **reverse input words** — a deliberately unnatural task (character-level transformations like reversing a word are not meaningful language modeling tasks) that serves as a lightweight benchmark for evaluating performance of Apple M-series devices for fine-tuning via MPS.

---

## 💻 Benchmarks

| Device           | Model                | avg it/s | eval_loss (5 Epochs) |
|------------------|----------------------|------------|------------------------|
| M2 Max, 30 Cores (32 GB)    | Qwen/Qwen3-0.6B-Base | 2.73       | 3.65                   |
| M3 Max, 40 Cores (128 GB)    | Qwen/Qwen3-0.6B-Base | 3.26       | 3.10                   |
| M3 Max, 40 Cores (128 GB)    | Qwen/Qwen3-4B-Base | 0.63       | 2.34                   |


> 🧪 **PRs welcome!** Add your own results (any model / any M-series device).

---

## How It Works

- **Input:** a single word (e.g., `apple`)
- **Target:** the reversed word with EOS (e.g., `elppa <eos>`)
- **Objective:** Teach the model to learn this transformation from scratch
