import torch
import torch.nn as nn
import torch.nn.functional as F


class ForensicAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads: int = 8,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        return x


class ForensicAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads: int = 8,
        qkv_bias: bool = True,
        mlp_ratio: float = 4,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ForensicAttention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ForensicAttentionOutput(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        x = F.normalize(x, p=2, dim=2)
        output = x @ x.transpose(-2, -1)
        return output


class ForensicAttentionHead(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads: int = 8,
        num_blocks: int = 4,
        qkv_bias: bool = True,
        mlp_ratio: float = 4,
    ):
        super().__init__()
        self.blocks = []
        for _ in range(num_blocks):
            self.blocks.append(
                ForensicAttentionBlock(
                    embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    mlp_ratio=mlp_ratio,
                )
            )
        self.blocks = nn.Sequential(*self.blocks)
        self.output = ForensicAttentionOutput(embed_dim)

    def forward(self, x):
        x = self.blocks(x)
        x = self.output(x)
        return x
