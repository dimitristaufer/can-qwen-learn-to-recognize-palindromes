import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1) device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# 2) load fine-tuned model
model_dir = "reverse-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).to(device)

# 3) inference function
def is_palindrome(word: str) -> str:
    # 1) build prompt with exactly one trailing space
    prompt = word.strip() + " "
    enc    = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    # 2) generate enough tokens for the label
    out = model.generate(
        **enc,
        max_new_tokens=10,
        do_sample=False,
        eos_token_id = tokenizer.eos_token_id,
        pad_token_id = tokenizer.eos_token_id,
    )

    # 3) token-level slicing
    input_len = enc.input_ids.size(-1)
    gen_ids   = out[0, input_len:]

    # 4) decode and return
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

# 4) example usage
if __name__ == "__main__":
    for w in ["racecar", "apple", "madam", "Dimitri", "Decision", "Bald"]:
        print(f"{w} → {is_palindrome(w)}")
