import pandas as pd
import random
import torch
from torch.utils.data import Dataset
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
    low_cpu_mem_usage=True,
).to(device)

#model.gradient_checkpointing_enable()

class ReverseDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=30):
        df = pd.read_csv(csv_path).dropna(subset=["word"])
        # build (prompt, reversed) pairs
        self.examples = [
            (str(w), str(w)[::-1])
            for w in df["word"]
        ]
        random.shuffle(self.examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prompt, rev = self.examples[idx]
        # tokenize prompt only to get its length
        p_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        p_len = len(p_ids)

        full = f"{prompt} {rev} {tokenizer.eos_token}" # add eos token
        enc  = self.tokenizer(
            full,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids      = enc.input_ids.squeeze(0).to(device)
        attention_mask = enc.attention_mask.squeeze(0).to(device)

        # labels: mask prompt & pad, predict only reversed tokens
        labels = input_ids.clone()
        labels[:p_len]            = -100
        labels[attention_mask==0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

train_dataset = ReverseDataset("palindromes_dataset_233_709.csv", tokenizer, max_length=30)

training_args = TrainingArguments(
    output_dir="./reverse-finetuned",
    per_device_train_batch_size=1,
    #gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=50,
    logging_steps=40,
    #save_steps=100,
    save_strategy="no",
    save_total_limit=1,
    fp16=False,
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./reverse-finetuned")
tokenizer.save_pretrained("./reverse-finetuned")