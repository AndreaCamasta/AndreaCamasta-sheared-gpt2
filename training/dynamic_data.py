import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import List, Dict, Optional

from datasets import load_dataset
from transformers import GPT2LMHeadModel


@dataclass
class DomainConfig:
    name: str
    hf_dataset: str
    subset: Optional[str]
    split: str
    text_field: str
    max_samples: int


def get_default_domain_configs(max_samples_per_domain: int = 10000) -> List[DomainConfig]:
    """
    3 domains using parquet-based datasets:

    - wiki: wikitext-2-raw-v1 (LM-style)
    - reviews: yelp_review_full (user reviews)
    - news: ag_news (news headlines + descriptions)
    """
    return [
        DomainConfig(
            name="wiki",
            hf_dataset="wikitext",
            subset="wikitext-2-raw-v1",
            split="train",
            text_field="text",
            max_samples=max_samples_per_domain,
        ),
        DomainConfig(
            name="reviews",
            hf_dataset="yelp_review_full",
            subset=None,
            split="train",
            text_field="text",
            max_samples=max_samples_per_domain,
        ),
        DomainConfig(
            name="news",
            hf_dataset="ag_news",
            subset=None,
            split="train",
            text_field="text",
            max_samples=max_samples_per_domain,
        ),
    ]


def _load_and_tokenize_domain(
    tokenizer,
    cfg: DomainConfig,
    max_length: int = 128,
):
    """
    Load a single domain from HF and tokenize with the given tokenizer.
    Returns a tokenized Dataset with 'input_ids' and 'attention_mask'.
    """
    print(f"Loading domain '{cfg.name}' from {cfg.hf_dataset} ({cfg.subset})...")
    if cfg.subset is not None:
        raw = load_dataset(cfg.hf_dataset, cfg.subset, split=cfg.split)
    else:
        raw = load_dataset(cfg.hf_dataset, split=cfg.split)

    # Subsample for practicality
    if cfg.max_samples is not None and cfg.max_samples < len(raw):
        raw = raw.select(range(cfg.max_samples))
        print(f"  Subsampled to {cfg.max_samples} examples.")

    def encode(batch):
        return tokenizer(
            batch[cfg.text_field],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = raw.map(
        encode,
        batched=True,
        remove_columns=[c for c in raw.column_names if c != cfg.text_field],
    )

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized


def load_multi_domain_datasets(
    tokenizer,
    domain_cfgs: List[DomainConfig],
    max_length: int = 128,
) -> Dict[str, "datasets.Dataset"]:
    """
    Returns: dict[name] -> tokenized HF Dataset
    """
    domain_datasets = {}
    for cfg in domain_cfgs:
        ds = _load_and_tokenize_domain(tokenizer, cfg, max_length=max_length)
        domain_datasets[cfg.name] = ds
        print(f"Domain '{cfg.name}' loaded with {len(ds)} examples.")
    return domain_datasets


def compute_reference_losses(
    domain_datasets: Dict[str, "datasets.Dataset"],
    tokenizer,
    base_model_name: str = "gpt2",
    device: str = "cuda",
    eval_samples: int = 512,
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Compute average LM loss per domain using the reference (unpruned) model.
    Returns: dict[name] -> avg_loss
    """
    print(f"Loading reference model '{base_model_name}' for dynamic batching...")
    ref_model = GPT2LMHeadModel.from_pretrained(base_model_name).to(device)
    ref_model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    ref_losses = {}

    for name, ds in domain_datasets.items():
        print(f"\nComputing reference loss for domain '{name}'...")
        n_eval = min(eval_samples, len(ds))
        ds_eval = ds.select(range(n_eval))

        loader = DataLoader(ds_eval, batch_size=batch_size)

        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = input_ids.clone()

                outputs = ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        ref_losses[name] = avg_loss
        print(f"  Reference loss for '{name}': {avg_loss:.4f}")

    return ref_losses
