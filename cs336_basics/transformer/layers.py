import torch
import torch.nn as nn
from torch import Tensor

from math import sqrt
from einops import einsum, reduce, rearrange
from jaxtyping import Float, Bool, Int


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


def softmax(x: torch.Tensor, dimension: int) -> torch.Tensor:

    x_exp = torch.exp(x - torch.max(input=x, dim=dimension, keepdim=True).values)
    Z = x_exp.sum(dim=dimension, keepdim=True)
    x_softmax = x_exp / Z
    return x_softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... values d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:

    d_k = Q.shape[-1]

    qk = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / sqrt(d_k)

    if mask is not None:
        qk = torch.where(mask == False, float("-inf"), qk)

    qk = softmax(x=qk, dimension=-1)

    assert V.shape[-2] == K.shape[-2], f"Number of values and keys do not match"
    # TODO: V is (... values d_v) but we use (... keys d_v) for einsum to work
    attn = einsum(qk, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return attn


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_theta: float = None,
        max_seq_len: int = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = self.d_v = self.d_model // self.num_heads
        assert self.d_k * num_heads == self.d_model

        if rope_theta is not None and max_seq_len is not None:
            self.enable_rope = True
            self.rope = RoPE(theta=rope_theta, d_k=self.d_k, max_seq_len=max_seq_len)
        else:
            self.enable_rope = False

        self.q_proj = Linear(
            in_features=self.d_model, out_features=self.num_heads * self.d_k
        )
        self.k_proj = Linear(
            in_features=self.d_model, out_features=self.num_heads * self.d_k
        )
        self.v_proj = Linear(
            in_features=self.d_model, out_features=self.num_heads * self.d_v
        )
        self.out_proj = Linear(
            in_features=self.num_heads * self.d_v, out_features=self.d_model
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):

        self.q_proj.load_state_dict({"W": state_dict["q_proj"]})
        self.k_proj.load_state_dict({"W": state_dict["k_proj"]})
        self.v_proj.load_state_dict({"W": state_dict["v_proj"]})
        self.out_proj.load_state_dict({"W": state_dict["out_proj"]})

    def forward(
        self, x: Float[Tensor, "... sequence_length d_in"], token_positions=None
    ) -> Float:

        seq_len = x.shape[-2]

        Q = rearrange(
            self.q_proj(x),
            "... seq_len (h d_k) -> ... h seq_len d_k",
            h=self.num_heads,
            d_k=self.d_k,
        )
        K = rearrange(
            self.k_proj(x),
            "... seq_len (h d_k) -> ... h seq_len d_k",
            h=self.num_heads,
            d_k=self.d_k,
        )
        V = rearrange(
            self.v_proj(x),
            "... seq_len (h d_v) -> ... h seq_len d_v",
            h=self.num_heads,
            d_v=self.d_v,
        )

        mask = torch.tril(torch.ones(size=(seq_len, seq_len))).bool()

        # Apply Rotary Position Encodings
        if self.enable_rope:
            assert token_positions is not None
            Q = self.rope(x=Q, token_positions=token_positions)
            K = self.rope(x=K, token_positions=token_positions)

        attn = rearrange(
            scaled_dot_product_attention(Q=Q, K=K, V=V, mask=mask),
            "... h seq_len d_v -> ... seq_len (h d_v)",
            h=self.num_heads,
            d_v=self.d_v,
        )
        attn = self.out_proj(attn)
        return attn


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float,
    ):

        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta

        self.causal_mha = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
        )

        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

        self.norm_attn = RMSNorm(d_model=d_model)
        self.norm_ffn = RMSNorm(d_model=d_model)

    def load_state_dict(self, state_dict, strict=True, assign=False):

        # load attention weights
        attn_state_dict = {
            "q_proj": state_dict["attn.q_proj.weight"],
            "k_proj": state_dict["attn.k_proj.weight"],
            "v_proj": state_dict["attn.v_proj.weight"],
            "out_proj": state_dict["attn.output_proj.weight"],
        }
        self.causal_mha.load_state_dict(attn_state_dict)

        # load rmsnorm weights
        self.norm_attn.load_state_dict({"gain": state_dict["ln1.weight"]})
        self.norm_ffn.load_state_dict({"gain": state_dict["ln2.weight"]})

        # load ffn weights
        self.ffn.load_state_dict(
            {
                "W1": state_dict["ffn.w1.weight"],
                "W2": state_dict["ffn.w2.weight"],
                "W3": state_dict["ffn.w3.weight"],
            }
        )

    def forward(
        self, x: Float[Tensor, "batch sequence_length d_model"]
    ) -> Float[Tensor, "batch sequence_length d_model"]:

        seq_len = x.shape[1]
        token_positions = torch.arange(start=0, end=seq_len).view(1, -1)

        x = x + self.causal_mha(x=self.norm_attn(x), token_positions=token_positions)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        context_length: int,
        rope_theta: float,
        vocab_size: int,
        num_layers: int,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.context_length = context_length
        self.rope_theta = rope_theta
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embeds_token = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer_blocks = nn.Sequential()
        for _ in range(num_layers):
            self.transformer_blocks.append(
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    rope_theta=rope_theta,
                )
            )

        self.norm = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

    def load_state_dict(self, state_dict, strict=True, assign=False):

        # load token embeddings
        self.embeds_token.load_state_dict(
            {"embed_weights": state_dict["token_embeddings.weight"]}
        )

        # load transformer blocks weights
        for layer in range(self.num_layers):
            block_state_dict = {}
            for key, val in state_dict.items():
                if key.startswith(f"layers.{layer}."):
                    block_state_dict[key.replace(f"layers.{layer}.", "")] = val

            self.transformer_blocks[layer].load_state_dict(block_state_dict)

        # final layer norm weight
        self.norm.load_state_dict({"gain": state_dict["ln_final.weight"]})

        # lm head weight
        self.lm_head.load_state_dict({"W": state_dict["lm_head.weight"]})

    def forward(self, in_indices: Int[Tensor, "batch_size sequence_length"]):

        x = self.embeds_token(in_indices)
        x = self.transformer_blocks(x)
        x = self.norm(x)
        x = self.lm_head(x)

        return x
