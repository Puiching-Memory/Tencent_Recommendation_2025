from dataclasses import dataclass
import os

@dataclass
class ModelArgs:
    dim: int = None
    embedding_dim: int = None # if None, dim should be take as embedding dim
    n_layers: int = None
    n_heads: int = None
    item_vocab_size: int = None
    cate_vocab_size: int = None
    segment_vocab_size: int = 2
    multiple_of: int = 32  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_seq_len: int = None
    dropout:int = None
    rank_loss_weight:float = 1.0
    use_causal_mask:bool=True
    # AR
    n_samples:int = 4096
    temperature:float = 0.05
    l2_norm:bool = True
    item_ar_loss_weight:float = 0.0
    cate_ar_loss_weight:float = 0.0
    # Other
    attention_type:str = "bilinear_attention"  # bilinear_attention, din_attention, None
    pos_emb_dim:int = 32

    def __post_init__(self):
        """Initialize from environment variables if not provided"""
        if self.dim is None:
            self.dim = int(os.environ.get("MODEL_DIM", 128))
        if self.n_layers is None:
            self.n_layers = int(os.environ.get("MODEL_N_LAYERS", 4))
        if self.n_heads is None:
            self.n_heads = int(os.environ.get("MODEL_N_HEADS", 8))
        if self.max_seq_len is None:
            self.max_seq_len = int(os.environ.get("MODEL_MAX_SEQ_LEN", 50))
        if self.dropout is None:
            self.dropout = float(os.environ.get("MODEL_DROPOUT", 0.1))
