import os
import csv
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, AutoTokenizer, get_linear_schedule_with_warmup

from training.dynamic_data import (
    get_default_domain_configs,
    load_multi_domain_datasets,
    compute_reference_losses,
)
from training.data_utils import lm_collate_fn


def run_dynamic_training(
    sheared_model_dir: str,
    base_model_name: str = "gpt2",
    batch_size: int = 2,
    max_length: int = 128,
    num_train_steps: int = 2000,
    lr: float = 5e-5,
    device: str = "cuda",
    checkpoint_dir: str = "./checkpoints_sheared_dynamic",
    save_every: int = 500,
    alpha: float = 0.05,
    gamma: float = 1.0,
    log_every: int = 10,   # NEW: log frequency
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
    log_path = os.path.join(checkpoint_dir, "domain_sampling_log.csv")

    # 1) Load sheared model + tokenizer
    print(f"Loading sheared model from: {sheared_model_dir}")
    model = GPT2LMHeadModel.from_pretrained(sheared_model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(sheared_model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model.train()

    # 2) Load domain datasets
    domain_cfgs = get_default_domain_configs(max_samples_per_domain=5000)
    domain_datasets = load_multi_domain_datasets(tokenizer, domain_cfgs, max_length=max_length)

    # 3) Dataloaders per domain
    domain_loaders, domain_iters = {}, {}
    for name, ds in domain_datasets.items():
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lm_collate_fn)
        domain_loaders[name] = loader
        domain_iters[name] = iter(loader)

    domain_names = list(domain_datasets.keys())

    # 4) Reference losses per domain
    ref_losses = compute_reference_losses(
        domain_datasets,
        tokenizer,
        base_model_name=base_model_name,
        device=device,
        eval_samples=512,
        batch_size=8,
    )

    # EMA ratios
    ema_ratios = {name: 1.0 for name in domain_names}

    # 5) Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_train_steps),
        num_training_steps=num_train_steps,
    )

    # 6) CSV logging setup
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            # columns: step, sampled_domain, lm_loss, ratio, p_* for each domain, Rhat_* for each domain
            header = ["step", "sampled_domain", "lm_loss", "ratio"]
            header += [f"p_{n}" for n in domain_names]
            header += [f"Rhat_{n}" for n in domain_names]
            writer.writerow(header)

    print("\nStarting dynamic training...")
    for step in range(num_train_steps):
        # probs from EMA ratios
        weights = torch.tensor([(ema_ratios[n] + 1e-4) ** gamma for n in domain_names], dtype=torch.float32)
        probs = weights / weights.sum()

        # sample a domain
        idx = torch.multinomial(probs, num_samples=1).item()
        dom_name = domain_names[idx]

        # batch
        try:
            batch = next(domain_iters[dom_name])
        except StopIteration:
            domain_iters[dom_name] = iter(domain_loaders[dom_name])
            batch = next(domain_iters[dom_name])

        batch = {k: v.to(device) for k, v in batch.items()}

        # forward
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        lm_loss = outputs.loss

        # ratio vs reference
        ratio = lm_loss.item() / (ref_losses[dom_name] + 1e-8)

        # EMA update
        ema_ratios[dom_name] = (1 - alpha) * ema_ratios[dom_name] + alpha * ratio

        # update
        lm_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # log to CSV periodically
        if step % log_every == 0:
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                row = [step, dom_name, float(lm_loss.item()), float(ratio)]
                row += [float(probs[i].item()) for i in range(len(domain_names))]
                row += [float(ema_ratios[n]) for n in domain_names]
                writer.writerow(row)

        # print occasionally
        if step % 100 == 0:
            prob_str = ", ".join(f"{n}:{probs[i].item():.2f}" for i, n in enumerate(domain_names))
            rhat_str = ", ".join(f"{n}:{ema_ratios[n]:.2f}" for n in domain_names)
            print(f"Step {step:5d} | dom={dom_name:7s} | LM loss={lm_loss.item():.3f} | ratio={ratio:.3f} | p={prob_str} | R_hat={rhat_str}")

        # checkpoint
        if (step + 1) % save_every == 0 or (step + 1) == num_train_steps:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "ema_ratios": ema_ratios,
                "ref_losses": ref_losses,
                "domain_names": domain_names,
                "log_path": log_path,
            }
            torch.save(ckpt, ckpt_path)
            print(f"[Checkpoint] Saved dynamic training at step {step} → {ckpt_path}")
            print(f"[Log] Appending sampling stats to → {log_path}")

    print("\nDynamic training finished.")
    model.eval()
    return model, tokenizer, ema_ratios, ref_losses, log_path
