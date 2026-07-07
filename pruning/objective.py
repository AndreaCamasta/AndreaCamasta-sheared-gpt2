import torch
from typing import Dict

from config import PruningSpec
from models.masked_gpt2 import MaskedGPT2LMHeadModel


def pruning_constraint_loss(
    model: MaskedGPT2LMHeadModel,
    spec: PruningSpec,
    lambdas: Dict[str, float],
) -> torch.Tensor:
    """
    Compute a constraint loss that encourages the expected number of
    active layers / heads / FFN neurons to match the target architecture.

    IMPORTANT: we use *live* masks (with gradients), not detached stats.
    """

    # Live masks with gradients
    z_layer = model.layer_mask()          # shape [n_layer]
    z_head = model.head_mask()            # shape [n_layer, n_head]
    z_ffn = model.ffn_mask()              # shape [n_layer, n_inner]

    # Expected counts (sums of mask values)
    L_hat = z_layer.sum()
    H_hat = z_head.sum()
    M_hat = z_ffn.sum()

    # Target counts
    L_target = float(spec.tgt_n_layer)
    H_target = float(spec.tgt_n_layer * spec.tgt_n_head)
    M_target = float(spec.tgt_n_layer * spec.tgt_n_inner)

    # Squared error penalties
    loss_layer = (L_hat - L_target) ** 2
    loss_head = (H_hat - H_target) ** 2
    loss_ffn = (M_hat - M_target) ** 2

    # Optional L1 sparsity encouraging lower masks overall
    l1_layer = z_layer.mean()
    l1_head = z_head.mean()
    l1_ffn = z_ffn.mean()

    # Combine with lambdas
    layer_coeff = lambdas.get("layer", 0.0)
    head_coeff = lambdas.get("head", 0.0)
    ffn_coeff = lambdas.get("ffn", 0.0)

    l1_layer_coeff = lambdas.get("l1_layer", 0.0)
    l1_head_coeff = lambdas.get("l1_head", 0.0)
    l1_ffn_coeff = lambdas.get("l1_ffn", 0.0)

    constraint = (
        layer_coeff * loss_layer
        + head_coeff * loss_head
        + ffn_coeff * loss_ffn
        + l1_layer_coeff * l1_layer
        + l1_head_coeff * l1_head
        + l1_ffn_coeff * l1_ffn
    )

    return constraint
