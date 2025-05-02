"""
Architectures borrowed from timm. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from torch.distributions import Categorical


 
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, depth):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.blocks = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)
        ])
        self.act = nn.Tanh()
 
    def forward(self, x):
        x = self.act(self.input(x))
        for block in self.blocks:
            x = self.act(block(x))
        x = self.output(x)        
        return x


class Linear(nn.Module):
    def __init__(self, in_features, out_features, std=None, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        if std is None:
            nn.init.xavier_uniform_(self.fc.weight)
        else:
            nn.init.normal_(self.fc.weight, std=std)
        if bias:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class CategorialHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.ln = nn.LayerNorm(in_features)
        self.fc = Linear(in_features, out_features)

    def forward(self, x):
        logits = self.fc(self.ln(x))
        return Categorical(logits=logits)

    def loss(self, input, target):
        """
        Compute the loss of target.
        Args:
            input: (..., in_features)
            target: (...)
        Returns:
            (...) tensor, loss
        """
        result = self(input)
        return -result.log_prob(target)

    def sample(self, input):
        """
        Sample outputs from the input.
        Args:
            input: (..., in_features)
        Returns:
            (...) tensor, sampled output
        """
        return self(input).sample()


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        is_causal: bool = False,
        is_rotary: bool = False
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.is_causal = is_causal

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=self.is_causal,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


pos_emb_std = 0.02


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        is_causal: bool = False,
        cross_attn_first: int = 0,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.cross_attn_first = cross_attn_first
        if cross_attn_first:
            self.attn_self = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                is_causal=False,
            )
            self.ls_self = LayerScale(dim, init_values=init_values)
            self.norm_self = norm_layer(dim)
            # this will be applied to the first cross_attn_first tokens

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            is_causal=is_causal,
        )  # when cross_attn_first > 0, this will only be applied after the first cross_attn_first tokens

        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cross_attn_first > 0:
            attn_out = self.ls_self(
                self.attn_self(self.norm_self(x[:, : self.cross_attn_first]))
            )
            x = torch.cat(
                [
                    (x[:, : self.cross_attn_first] + attn_out),
                    x[:, self.cross_attn_first :],
                ],
                dim=1,
            )

        x = torch.cat(
            [
                x[:, : self.cross_attn_first],
                x[:, self.cross_attn_first :]
                + self.ls1(self.attn(self.norm1(x)))[
                    :, self.cross_attn_first :
                ],
            ],
            dim=1,
        )

        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        seq_len: int = 512,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        is_causal: bool = False,
        cross_attn_first: int = 0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.cross_attn_first = cross_attn_first
        self.output_ln = nn.LayerNorm(dim)
        self.input_ln1 = nn.LayerNorm(dim)
        self.input_ln2 = nn.LayerNorm(dim)
        self.input_scale = LayerScale(dim, init_values=pos_emb_std)

        self.pos_embed = nn.Parameter(torch.zeros(self.seq_len, dim))

        self.layers = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    proj_drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    mlp_layer=mlp_layer,
                    is_causal=is_causal,
                    init_values=1.0 / (num_layers**0.5),
                    cross_attn_first=cross_attn_first,
                )
                for i in range(num_layers)
            ]
        )

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.pos_embed, std=pos_emb_std)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            pass
            # we use xavier_uniform following official JAX ViT:
            # torch.nn.init.xavier_uniform_(m.weight)
            # if isinstance(m, nn.Linear) and m.bias is not None:
            # nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(
        self, x: torch.Tensor, pos_idx_lists: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass of the transformer model.
        Args:
            x: torch.Tensor: input tensor of shape (B, N, C)
        Returns:
            torch.Tensor: output tensor of shape (B, N, C)
        """
        B, N, C = x.shape
        x = self.input_ln1(x)
        x = self.input_scale(x)

        if pos_idx_lists is None:
            pos_idx_lists = [torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)]

        for pos_idx in pos_idx_lists:
            assert pos_idx.shape == (B, N)
            x = x + self.pos_embed[pos_idx]
        x = self.input_ln2(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_ln(x)
        return x

class Model(nn.Module):
    def __init__(self,
                 num_layers: int,
                 dim: int,
                 num_heads: int,
                 input_seq_len: int,
                 output_seq_len: int,
                 vocab_size: int):
        super().__init__()
        self.seq_len = max(input_seq_len, output_seq_len)
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.transormer = Transformer(
            num_layers=num_layers,
            dim=dim,
            num_heads=num_heads,
            seq_len=self.seq_len,
        )
        self.embedding = nn.Linear(vocab_size, dim)
        self.head = CategorialHead(dim, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = torch.zeros(x.shape[0], self.seq_len - self.input_seq_len, x.shape[2], device=x.device).long()
        x = torch.cat([x, pad], dim=1)
        x = self.embedding(x)
        x = self.transormer(x)
        x = self.head(x[:, :self.output_seq_len, :])
        return x
