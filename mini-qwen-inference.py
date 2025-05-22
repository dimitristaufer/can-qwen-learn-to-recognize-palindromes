import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "mps" if torch.backends.mps.is_available() else "cpu"

model_dir = "reverse-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).to(device)

def is_palindrome(word: str) -> str:

    prompt = word.strip() + " "
    enc    = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    out = model.generate(
        **enc,
        max_new_tokens=10,
        do_sample=False,
        eos_token_id = tokenizer.eos_token_id,
        pad_token_id = tokenizer.eos_token_id,
    )

    input_len = enc.input_ids.size(-1)
    gen_ids   = out[0, input_len:]

    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

for w in ["racecar", "apple", "madam", "dimitri", "decision", "bald"]:
        print(f"{w} → {is_palindrome(w)}")
