import torch
import torch.nn as nn

from einops import einsum, reduce


class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        weight = torch.empty(
            size=(out_features, in_features),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        weight = nn.init.trunc_normal_(
            tensor=weight,
            mean=0,
            std=2.0 / (out_features + in_features),
            a=-3,
            b=3,
        )
        self.W = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            x, self.W, "... in_features, out_features in_features -> ... out_features"
        )


class Embedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):

        super().__init__()

        embed_weights = torch.empty(
            size=(num_embeddings, embedding_dim), dtype=dtype, device=device
        )
        embed_weights = nn.init.trunc_normal_(embed_weights, mean=0, std=1, a=-3, b=+3)
        self.embed_weights = nn.Parameter(embed_weights, requires_grad=True)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_weights[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.gain = nn.Parameter(
            torch.ones(size=(self.d_model,), dtype=dtype, device=device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch_size, sequence_length, d_model) tensor"""

        dtype_orig = x.dtype
        x = x.to(torch.float32)  # upcasting to avoid overflow during squaring

        rms = torch.sqrt(x.square().sum(dim=-1, keepdim=True) / self.d_model + self.eps)
        rms_norm = x * self.gain / rms
        rms_norm = rms_norm.to(dtype=dtype_orig)
        return rms_norm
