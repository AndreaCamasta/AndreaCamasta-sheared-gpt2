import torch
from transformers import GPT2Config, GPT2LMHeadModel

from config import PruningSpec


def select_top_layers(masked_model, spec: PruningSpec):
    """
    Select which layers to keep based on learned layer_mask.
    Returns sorted indices of layers to keep.
    """
    z_layer = masked_model.layer_mask.expected_mask.detach().cpu()  # [n_layer]
    n_keep = spec.tgt_n_layer

    sorted_idx = torch.argsort(z_layer, descending=True)
    keep_idx = sorted_idx[:n_keep]
    keep_idx, _ = torch.sort(keep_idx)
    return keep_idx


def select_ffn(masked_model, spec: PruningSpec, keep_layers):
    """
    For each kept layer, choose which FFN neurons to keep
    based on ffn_mask. Heads are NOT structurally pruned here.
    """
    z_ffn = masked_model.ffn_mask.expected_mask.detach().cpu()    # [L, M]

    src_n_inner = z_ffn.shape[1]
    tgt_n_inner = spec.tgt_n_inner

    ffn_to_keep = {}

    for li in keep_layers:
        li = int(li.item())
        ffn_scores = z_ffn[li]  # [M]
        if tgt_n_inner >= src_n_inner:
            keep_m = torch.arange(src_n_inner)
        else:
            top_m = torch.argsort(ffn_scores, descending=True)[:tgt_n_inner]
            keep_m, _ = torch.sort(top_m)
        ffn_to_keep[li] = keep_m
    return ffn_to_keep


def build_sheared_gpt2_layers_ffn(masked_model, spec: PruningSpec):
    """
    Build a smaller GPT-2 LMHeadModel that:
      - Keeps only top-k layers (layer_mask)
      - Within each kept layer, keeps only top FFN neurons (ffn_mask)
      - Keeps ALL attention heads structurally (no head pruning)

    All weight slicing/copying is done on CPU to avoid CUDA indexing issues.
    """
    device = next(masked_model.parameters()).device
    big_model = masked_model.gpt2
    big_cfg = big_model.config

    # 1) Select layers
    keep_layers = select_top_layers(masked_model, spec)
    n_keep = len(keep_layers)

    # 2) Select FFN units per kept layer
    ffn_to_keep = select_ffn(masked_model, spec, keep_layers)

    # 3) Build new config: fewer layers, smaller FFN, same n_head / n_embd
    cfg_dict = big_cfg.to_dict()
    cfg_dict["n_layer"] = int(n_keep)
    cfg_dict["n_inner"] = int(spec.tgt_n_inner)
    cfg_dict["n_head"] = big_cfg.n_head
    cfg_dict["n_embd"] = big_cfg.n_embd

    small_cfg = GPT2Config(**cfg_dict)

    # 4) Initialize the small model on CPU
    small_model = GPT2LMHeadModel(small_cfg)

    # 5) Copy shared components (embeddings, ln_f, lm_head) on CPU
    with torch.no_grad():
        # Token embeddings
        small_model.transformer.wte.load_state_dict(
            {k: v.detach().cpu() for k, v in big_model.transformer.wte.state_dict().items()}
        )
        # Positional embeddings
        small_model.transformer.wpe.load_state_dict(
            {k: v.detach().cpu() for k, v in big_model.transformer.wpe.state_dict().items()}
        )
        # Final layer norm
        small_model.transformer.ln_f.load_state_dict(
            {k: v.detach().cpu() for k, v in big_model.transformer.ln_f.state_dict().items()}
        )
        # LM head
        small_model.lm_head.load_state_dict(
            {k: v.detach().cpu() for k, v in big_model.lm_head.state_dict().items()}
        )

        # 6) For each kept layer: copy attention as-is, prune MLP
        for new_i, old_i in enumerate(keep_layers):
            old_i = int(old_i.item())
            old_block = big_model.transformer.h[old_i]
            new_block = small_model.transformer.h[new_i]

            # --- Attention: copy weights directly (no head pruning) ---
            new_block.attn.load_state_dict(
                {k: v.detach().cpu() for k, v in old_block.attn.state_dict().items()}
            )

            # --- MLP: prune FFN neurons by slicing weights ---
            old_mlp = old_block.mlp
            new_mlp = new_block.mlp

            keep_m = ffn_to_keep[old_i]  # indices in [0, src_n_inner)

            # c_fc.weight: [n_embd, n_inner]  (H, M)
            # c_fc.bias:   [n_inner]
            old_c_fc_w = old_mlp.c_fc.weight.detach().cpu()
            old_c_fc_b = old_mlp.c_fc.bias.detach().cpu()

            # new_mlp.c_fc.weight: [n_embd, tgt_n_inner]
            new_mlp.c_fc.weight.copy_(old_c_fc_w[:, keep_m])
            new_mlp.c_fc.bias.copy_(old_c_fc_b[keep_m])

            # c_proj.weight: [n_inner, n_embd]  (M, H)
            # c_proj.bias:   [n_embd]
            old_c_proj_w = old_mlp.c_proj.weight.detach().cpu()
            old_c_proj_b = old_mlp.c_proj.bias.detach().cpu()

            # new_mlp.c_proj.weight: [tgt_n_inner, n_embd]
            new_mlp.c_proj.weight.copy_(old_c_proj_w[keep_m, :])
            new_mlp.c_proj.bias.copy_(old_c_proj_b)

            # --- Layer norms ---
            new_block.ln_1.load_state_dict(
                {k: v.detach().cpu() for k, v in old_block.ln_1.state_dict().items()}
            )
            new_block.ln_2.load_state_dict(
                {k: v.detach().cpu() for k, v in old_block.ln_2.state_dict().items()}
            )

    # Finally move small model back to original device (GPU/CPU)
    small_model = small_model.to(device)
    return small_model, keep_layers, ffn_to_keep
