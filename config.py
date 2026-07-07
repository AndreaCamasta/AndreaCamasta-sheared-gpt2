from dataclasses import dataclass
from transformers import GPT2Config

@dataclass
class PruningSpec:
    """
    Describes the source and target architecture for pruning.
    """
    # source model sizes
    src_n_layer: int
    src_n_head: int
    src_n_inner: int
    src_n_embd: int

    # target sizes (fixed beforehand)
    tgt_n_layer: int
    tgt_n_head: int
    tgt_n_inner: int
    tgt_n_embd: int

    @property
    def layer_compression(self) -> float:
        return self.tgt_n_layer / self.src_n_layer

    @property
    def head_compression(self) -> float:
        return self.tgt_n_head / self.src_n_head

    @property
    def ffn_compression(self) -> float:
        return self.tgt_n_inner / self.src_n_inner

    @property
    def embd_compression(self) -> float:
        return self.tgt_n_embd / self.src_n_embd


def make_default_pruning_spec(model_name: str = "gpt2") -> PruningSpec:
    """
    Build a default PruningSpec for a given GPT-2 variant.

    For now we target ~50% compression on layers, heads, and FFN width
    while keeping hidden size the same.
    """
    cfg = GPT2Config.from_pretrained(model_name)

    src_n_layer = cfg.n_layer
    src_n_head = cfg.n_head
    # n_inner might be None for some configs; default is 4 * n_embd
    src_n_inner = cfg.n_inner or (4 * cfg.n_embd)
    src_n_embd = cfg.n_embd

    # Example: for gpt2 (12L, 12H, 3072 FFN, 768 embd) → target ~half
    if model_name == "gpt2":
        tgt_n_layer = 6
        tgt_n_head = 6
        tgt_n_inner = 1536
        tgt_n_embd = src_n_embd  # keep hidden size fixed
    else:
        # generic ~50% for other GPT-2 variants
        tgt_n_layer = max(1, src_n_layer // 2)
        tgt_n_head = max(1, src_n_head // 2)
        tgt_n_inner = max(4 * 16, src_n_inner // 2)
        tgt_n_embd = src_n_embd  # hidden-size pruning = future work

    return PruningSpec(
        src_n_layer=src_n_layer,
        src_n_head=src_n_head,
        src_n_inner=src_n_inner,
        src_n_embd=src_n_embd,
        tgt_n_layer=tgt_n_layer,
        tgt_n_head=tgt_n_head,
        tgt_n_inner=tgt_n_inner,
        tgt_n_embd=tgt_n_embd,
    )
