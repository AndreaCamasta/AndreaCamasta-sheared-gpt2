import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from typing import Optional, Dict

from config import PruningSpec, make_default_pruning_spec
from pruning.masks import HardConcreteMask


class MaskedGPT2LMHeadModel(nn.Module):
    """
    Wrapper around GPT2LMHeadModel that adds learnable structured masks
    (layers, heads, FFN neurons) and applies the layer mask inside the
    transformer via forward hooks.
    """

    def __init__(
        self,
        base_model_name: str = "gpt2",
        pruning_spec: Optional[PruningSpec] = None,
    ):
        super().__init__()
        self.base_model_name = base_model_name

        if pruning_spec is None:
            pruning_spec = make_default_pruning_spec(base_model_name)
        self.pruning_spec = pruning_spec

        # Load the pretrained GPT-2 LM head model
        self.gpt2 = GPT2LMHeadModel.from_pretrained(base_model_name)

        n_layer = self.gpt2.config.n_layer
        n_head = self.gpt2.config.n_head
        n_inner = self.gpt2.config.n_inner or (4 * self.gpt2.config.n_embd)

        # Masks (initialized around 0.5 instead of ~1.0)
        init_log_alpha = 0.0  # ~sigmoid(0) = 0.5 before stretching/clamping
        self.layer_mask = HardConcreteMask((n_layer,), init_log_alpha=init_log_alpha)
        self.head_mask = HardConcreteMask((n_layer, n_head), init_log_alpha=init_log_alpha)
        self.ffn_mask = HardConcreteMask((n_layer, n_inner), init_log_alpha=init_log_alpha)

        # Storage for current layer mask during a forward pass
        self._current_z_layer = None

        # Register forward hooks on each transformer block to apply layer mask
        self._register_layer_hooks()

    def _register_layer_hooks(self):
        """
        Attach a forward hook to each GPT-2 block that scales its output
        by the corresponding layer mask value z_layer[layer_idx].
        """
        self._hooks = []
        for layer_idx, block in enumerate(self.gpt2.transformer.h):

            def make_hook(i):
                def hook(module, input, output):
                    z_layer = self._current_z_layer
                    if z_layer is None:
                        return output

                    # scalar for this layer
                    z = z_layer[i]  # shape []
                    # We may get a tuple: (hidden_states, present, ...)
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                        # Broadcast z to match hidden_states dims
                        z_b = z
                        while z_b.dim() < hidden_states.dim():
                            z_b = z_b.unsqueeze(0)
                        hidden_states = hidden_states * z_b
                        # Rebuild tuple with scaled hidden_states
                        return (hidden_states,) + output[1:]
                    else:
                        # Single tensor case
                        z_b = z
                        while z_b.dim() < output.dim():
                            z_b = z_b.unsqueeze(0)
                        return output * z_b

                return hook

            h = block.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(h)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        """
        Compute a forward pass where each transformer block's output is
        scaled by its layer mask value.
        """
        # Sample / compute the current layer mask for this forward pass
        z_layer = self.layer_mask()
        self._current_z_layer = z_layer

        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        # Clear for safety
        self._current_z_layer = None
        return outputs

    def get_mask_stats(self) -> Dict[str, torch.Tensor]:
        """
        Utility to retrieve current expected masks for analysis / pruning.
        (Detaches from graph; use only for reporting.)
        """
        return {
            "layer_mask": self.layer_mask.expected_mask.detach().cpu(),
            "head_mask": self.head_mask.expected_mask.detach().cpu(),
            "ffn_mask": self.ffn_mask.expected_mask.detach().cpu(),
        }
