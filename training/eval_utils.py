import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

def count_params(model):
    """Return parameter count (int)."""
    return sum(p.numel() for p in model.parameters())

def compute_perplexity(model, tokenizer, max_samples=200, max_length=128, batch_size=4, device="cuda"):
    """
    Compute perplexity on a small subset of wikitext-2 validation.
    This is not a full benchmark, but enough for comparison.
    """
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    def encode(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    tokenized = dataset.map(encode, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    dataloader = DataLoader(tokenized, batch_size=batch_size)

    total_loss = 0
    total_tokens = 0
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="sum")

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # HF returns mean loss; we need sum of log-likelihoods
            loss = outputs.loss * input_ids.numel() / batch["input_ids"].shape[0]
            total_loss += loss.item()
            total_tokens += attention_mask.sum().item()

    ppl = torch.exp(torch.tensor(total_loss / total_tokens))
    return ppl.item()


def benchmark_generation(model, tokenizer, device="cuda", prompt="The future of AI is", max_new_tokens=50):
    """
    Measure tokens per second for greedy generation.
    Simple but effective benchmark.
    """
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warm-up
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10)

    # Benchmark
    import time
    start = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    end = time.time()

    elapsed = end - start
    tokens_generated = max_new_tokens
    tps = tokens_generated / elapsed

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return tps, text
