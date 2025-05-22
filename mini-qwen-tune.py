import pandas as pd
import random
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype  = torch.float32   # NO FP16 for training on MPS, as it creates NANs

model_name = "Qwen/Qwen3-0.6B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=dtype,
    #low_cpu_mem_usage=True, # enable for less memory usage
).to(device)

#model.gradient_checkpointing_enable() # enable for less memory usage

class ReverseDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=30, device="cpu"):
        df = pd.read_csv(csv_path).dropna(subset=["word"])
        self.examples = [(w, w[::-1]) for w in df["word"].astype(str)]
        random.shuffle(self.examples)

        self.tk = tokenizer
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prompt, rev = self.examples[idx]

        prompt_ids = self.tk(prompt, add_special_tokens=False).input_ids # prompt tokenized
        rev_ids    = self.tk(rev,    add_special_tokens=False).input_ids # reversed word tokenized

        full_ids = prompt_ids + rev_ids + [self.eos_id] # token ids of prompt + reversed + EOS

        seq_len = len(full_ids)
        if seq_len > self.max_length:
            full_ids = full_ids[:self.max_length] # truncate if too long
            attention_mask = [1]*self.max_length
        else:
            attention_mask = [1]*seq_len + [0]*(self.max_length - seq_len)
            full_ids       = full_ids + [self.pad_id]*(self.max_length - seq_len)

        input_ids = torch.tensor(full_ids, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(self.device)

        labels = input_ids.clone()
        labels[:len(prompt_ids)] = -100 # mask prompt tokens, leave reversed tokens and EOS unmasked
        labels[attention_mask == 0] = -100 # mask the padding

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

full_dataset = ReverseDataset("palindromes_dataset_233_709.csv", tokenizer, max_length=30, device=device)

eval_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - eval_size
train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size]) # split 90 / 10

training_args = TrainingArguments(
    output_dir="./reverse-finetuned",
    per_device_train_batch_size=1,
    #gradient_accumulation_steps=2, # enable for less memory usage
    num_train_epochs=5,
    learning_rate=3e-6,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=80,
    save_strategy="no",
    save_total_limit=1,
    evaluation_strategy="epoch",
    fp16=False,
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset, # 90 %
    eval_dataset=eval_dataset, # 10 %
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./reverse-finetuned")
tokenizer.save_pretrained("./reverse-finetuned")