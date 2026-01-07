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


def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dtype=None, device=None):

        super().__init__()

        self.W1 = self.__get_linear_weights__(
            in_features=d_ff, out_features=d_model, dtype=dtype, device=device
        )
        self.W2 = self.__get_linear_weights__(
            in_features=d_model, out_features=d_ff, dtype=dtype, device=device
        )
        self.W3 = self.__get_linear_weights__(
            in_features=d_ff, out_features=d_model, dtype=dtype, device=device
        )

    def __get_linear_weights__(
        self, in_features: int, out_features: int, dtype=None, device=None
    ):
        weights = nn.init.trunc_normal_(
            torch.empty(size=(in_features, out_features), dtype=dtype, device=device),
            mean=0,
            std=2.0 / (in_features + out_features),
        )
        return nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x1 = SiLU(einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff"))
        x2 = x1 * einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")

        result = einsum(x2, self.W2, "... d_ff, d_model d_ff -> ... d_model")
        return result


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        self.__precompute_rotation_matrices()

    def __precompute_rotation_matrices(self):
        """
        Note:
        RoPE rotates pairs of elements in the input vectors by angles that depend on:
        - The position of the token in the sequence (i)
        - Which pair of dimension we are rotating (k)
        """

        i_arr = torch.arange(self.max_seq_len, device=self.device).view(
            -1, 1
        )  # (max_seq_len, 1)

        k_arr = torch.arange(1, self.d_k // 2 + 1, device=self.device).view(
            1, -1
        )  # (1, d_k / 2)

        theta_ik = i_arr / (
            self.theta ** ((2 * k_arr - 2) / self.d_k)
        )  # (max_seq_len, d_k / 2)

        cos_ik = torch.cos(theta_ik)  # (max_seq_len, d_k / 2)
        sin_ik = torch.sin(theta_ik)  # (max_seq_len, d_k / 2)

        self.register_buffer("cos_ik", cos_ik, persistent=False)
        self.register_buffer("sin_ik", sin_ik, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x:                (..., seq_len, d_k)
        # token_positions:  (..., seq_len) -> token positions of x along the sequence dimension

        """
        - Why are token_positions required instead of just using range(0, seq_len)?
            - There might be packing of samples, which breaks the range assumption
        """

        cos_vals = self.cos_ik[token_positions]  # (..., seq_len, d_k / 2)
        sin_vals = self.sin_ik[token_positions]  # (..., seq_len, d_k / 2)

        idxs_even = torch.arange(start=0, end=self.d_k, step=2, dtype=torch.long)
        idxs_odd = torch.arange(start=1, end=self.d_k, step=2, dtype=torch.long)

        x_even = x[..., idxs_even]  # (..., seq_len, d_k / 2)
        x_odd = x[..., idxs_odd]  # (..., seq_len, d_k / 2)

        rotated_even = x_even * cos_vals - x_odd * sin_vals
        rotated_odd = x_even * sin_vals + x_odd * cos_vals

        x[..., idxs_even] = rotated_even
        x[..., idxs_odd] = rotated_odd

        return x
