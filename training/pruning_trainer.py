import os
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from config import make_default_pruning_spec
from models.masked_gpt2 import MaskedGPT2LMHeadModel
from pruning.objective import pruning_constraint_loss
from training.data_utils import load_tiny_lm_dataset, lm_collate_fn


def run_pruning_demo(
    model_name: str = "gpt2",
    batch_size: int = 4,
    max_length: int = 128,
    num_train_steps: int = 100,
    lr: float = 5e-5,
    device: str = "cuda",
):
    spec = make_default_pruning_spec(model_name)
    print("PruningSpec:", spec)

    dataset, tokenizer = load_tiny_lm_dataset(
        model_name=model_name,
        max_length=max_length,
        num_samples=2048,
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=lm_collate_fn
    )

    model = MaskedGPT2LMHeadModel(base_model_name=model_name, pruning_spec=spec)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = num_train_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    lambdas = {
        "layer": 1e-2,
        "head": 1e-5,
        "ffn": 1e-7,
        "l1_layer": 0.0,
        "l1_head": 0.0,
        "l1_ffn": 0.0,
    }

    step = 0
    data_iter = iter(dataloader)

    while step < num_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        lm_loss = outputs.loss

        constr_loss = pruning_constraint_loss(model, spec, lambdas)
        loss = lm_loss + constr_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 10 == 0:
            print(
                f"Step {step:4d} | "
                f"LM loss: {lm_loss.item():.3f} | "
                f"Constraint loss: {constr_loss.item():.3f} | "
                f"Total: {loss.item():.3f}"
            )

        step += 1

    stats = model.get_mask_stats()
    z_layer = stats["layer_mask"]
    z_head = stats["head_mask"]
    z_ffn = stats["ffn_mask"]

    L_hat = z_layer.sum().item()
    H_hat = z_head.sum().item()
    M_hat = z_ffn.sum().item()

    print("\nFinal expected counts (approx):")
    print(f"  Layers: {L_hat:.2f} (target {spec.tgt_n_layer})")
    print(f"  Heads:  {H_hat:.2f} (target {spec.tgt_n_layer * spec.tgt_n_head})")
    print(f"  FFN:    {M_hat:.2f} (target {spec.tgt_n_layer * spec.tgt_n_inner})")

    return model, spec, tokenizer


def run_pruning_with_checkpoints(
    model_name: str = "gpt2",
    batch_size: int = 4,
    max_length: int = 128,
    num_train_steps: int = 5000,
    lr: float = 3e-5,
    device: str = "cuda",
    checkpoint_dir: str = "./checkpoints_gpt2_pruning",
    save_every: int = 500,
    resume: bool = True,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "latest.pt")

    spec = make_default_pruning_spec(model_name)
    print("PruningSpec:", spec)

    dataset, tokenizer = load_tiny_lm_dataset(
        model_name=model_name,
        max_length=max_length,
        num_samples=8192,
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=lm_collate_fn
    )

    model = MaskedGPT2LMHeadModel(base_model_name=model_name, pruning_spec=spec)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = num_train_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    start_step = 0
    if resume and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt.get("step", 0) + 1
        print(f"Resuming from step {start_step}")

    if start_step >= num_train_steps:
        print("Checkpoint already at or beyond requested num_train_steps. Nothing to do.")
        model.eval()
        return model, spec, tokenizer

    # 🔴 Updated lambdas: stronger FFN pruning from now on
    lambdas = {
        "layer": 1e-2,
        "head": 1e-5,
        "ffn": 5e-7,   # <--- was 1e-7 before
        "l1_layer": 0.0,
        "l1_head": 0.0,
        "l1_ffn": 0.0,
    }

    step = start_step
    data_iter = iter(dataloader)

    print(f"Starting pruning run from step {step} to {num_train_steps}...")
    while step < num_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        lm_loss = outputs.loss

        constr_loss = pruning_constraint_loss(model, spec, lambdas)
        loss = lm_loss + constr_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            print(
                f"Step {step:6d} | "
                f"LM loss: {lm_loss.item():.3f} | "
                f"Constraint loss: {constr_loss.item():.3f} | "
                f"Total: {loss.item():.3f}"
            )

        if (step + 1) % save_every == 0 or (step + 1) == num_train_steps:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "spec_model_name": model_name,
            }
            torch.save(ckpt, ckpt_path)
            print(f"[Checkpoint] Saved at step {step} → {ckpt_path}")

        step += 1

    model.eval()
    stats = model.get_mask_stats()
    z_layer = stats["layer_mask"]
    z_head = stats["head_mask"]
    z_ffn = stats["ffn_mask"]

    L_hat = z_layer.sum().item()
    H_hat = z_head.sum().item()
    M_hat = z_ffn.sum().item()

    print("\nFinal expected counts (approx):")
    print(f"  Layers: {L_hat:.2f} (target {spec.tgt_n_layer})")
    print(f"  Heads:  {H_hat:.2f} (target {spec.tgt_n_layer * spec.tgt_n_head})")
    print(f"  FFN:    {M_hat:.2f} (target {spec.tgt_n_layer * spec.tgt_n_inner})")

    return model, spec, tokenizer
