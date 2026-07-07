from datasets import load_dataset
from transformers import AutoTokenizer
import torch

def load_tiny_lm_dataset(
    model_name: str = "gpt2",
    split: str = "train",
    max_length: int = 128,
    num_samples: int = 2048,
):
    """
    Load a small language modeling dataset and tokenize it for GPT-2.
    We use wikitext-2-raw-v1 as an example and subsample num_samples sequences.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    ds = ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized, tokenizer


def lm_collate_fn(batch):
    """
    Simple collate: stack input_ids and attention_mask and use input as labels.
    """
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
