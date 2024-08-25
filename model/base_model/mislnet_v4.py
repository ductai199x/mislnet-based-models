import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
from .base_model_class import BaseModel
from typing import *


act_mapping = {
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}

def get_activation_layer(activation: str):
    return act_mapping[activation.lower()]


def greatest_divisor(num: int, max_divisor: int):
    for i in range(max_divisor, 1, -1):
        if num % i == 0:
            return i
    return 1


class FusedConstrainedConv(nn.Module):
    def __init__(
        self,
        num_constrained_filters=32,
        num_rgb_filters=32,
        input_chan=3,
    ):
        super().__init__()
        self.kernel_size = 5
        self.input_chan = input_chan
        self.num_constrained_filters = num_constrained_filters
        self.num_rgb_filters = num_rgb_filters
        self.output_chan = num_constrained_filters + num_rgb_filters

        weight = torch.empty(num_constrained_filters, input_chan, self.kernel_size, self.kernel_size)
        nn.init.xavier_normal_(weight, gain=1/3)
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.one_middle = torch.zeros(self.kernel_size * self.kernel_size)
        self.one_middle[self.one_middle.numel() // 2] = 1
        self.one_middle = nn.Parameter(self.one_middle, requires_grad=False)

        self.rgb_weight = nn.Parameter(
            nn.init.xavier_normal_(
                torch.empty(num_rgb_filters, input_chan, self.kernel_size, self.kernel_size), gain=1 / 3
            ),
            requires_grad=True,
        )

    def forward(self, x):
        w_cstr = self.weight
        w_cstr = w_cstr.view(-1, self.kernel_size * self.kernel_size)
        w_cstr = w_cstr - w_cstr.mean(1)[..., None] + 1 / (self.kernel_size * self.kernel_size - 1)
        w_cstr = w_cstr - (w_cstr + 1) * self.one_middle
        w_cstr = w_cstr.view(
            self.num_constrained_filters, self.input_chan, self.kernel_size, self.kernel_size
        )
        weight = torch.cat([w_cstr, self.rgb_weight], dim=0)
        x = nn.functional.conv2d(x, weight, padding="same")
        return x


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
        self.attn = ForensicAttention(
            embed_dim, num_heads=num_heads, qkv_bias=qkv_bias)
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


class FusedConstrainedConvWithSelfAttention(nn.Module):
    def __init__(
        self,
        num_constrained_filters: int = 32,
        num_rgb_filters: int = 32,
        input_chan: int = 3,
        patch_size: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4,
        num_blocks: int = 4,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.fused_constrained_conv = FusedConstrainedConv(num_constrained_filters, num_rgb_filters, input_chan)
        self.weight = self.fused_constrained_conv.weight # This is to make it possible for lightning module class to access this weight
        self.output_chan = num_constrained_filters + num_rgb_filters
        embed_dim = (patch_size // 8) ** 2

        self.pre_proj = nn.Sequential(
            nn.Conv2d(self.output_chan, self.output_chan, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(self.output_chan, self.output_chan, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(self.output_chan, self.output_chan, 3, 2, 1),
        )
        self.post_proj = nn.Sequential(
            nn.Linear(embed_dim, 8 * 8),
            nn.GELU(),
            nn.Linear(8 * 8, 1),
        )
        self.self_attention_blocks = []
        for _ in range(num_blocks):
            self.self_attention_blocks.append(
                ForensicAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
            )
        self.self_attention_blocks = nn.Sequential(*self.self_attention_blocks)

    def forward(self, x):
        x = self.fused_constrained_conv(x)
        shortcut = x
        x = self.pre_proj(x)
        x = x.flatten(2, -1)
        x = self.self_attention_blocks(x)
        x = self.post_proj(x).permute(0, 2, 1).unsqueeze(-1)
        x = x * shortcut
        return x


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size=3,
        stride=1,
        padding=0,
    ):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_features,
            in_features,
            kernel_size,
            stride,
            padding,
            groups=in_features,
            bias=False,
        )
        self.pointwise_conv = nn.Conv2d(in_features, out_features, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class ConvBnAct(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size=3,
        stride=1,
        padding=0,
        act=True,
        use_separable_conv=False,
    ):
        super().__init__()
        if use_separable_conv and stride > 1:
            self.conv = SeparableConv2d(in_features, out_features, kernel_size, stride, padding)
        else:
            self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride, padding)
        if out_features % 16 != 0:
            num_groups = greatest_divisor(out_features, 16)
            print(f"WARNING: The number of output features {out_features} is not divisible by 16. Using {num_groups} groups.")
        else:
            num_groups = 16
        self.bn = nn.GroupNorm(num_groups, out_features)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size=3,
        stride=1,
        padding=0,
        expansion_factor=4,
        use_separable_conv=False,
    ):
        super().__init__()
        expanded_dim = int(expansion_factor * in_features)
        self.use_res_connect = stride == 1 and in_features == out_features

        self.fused_conv_block = ConvBnAct(
            in_features, expanded_dim, kernel_size, stride, padding, True, use_separable_conv
        )
        if in_features == out_features and expansion_factor == 1:
            self.conv1x1_block = nn.Identity()
        else:
            self.conv1x1_block = ConvBnAct(expanded_dim, out_features, 1, 1, 0, False, False)
        self.stochastic_depth = StochasticDepth(0.0, "row")

    def forward(self, x):
        residual = x
        output = self.fused_conv_block(x)
        output = self.conv1x1_block(output)
        if self.use_res_connect:
            output = self.stochastic_depth(output)
            output += residual
        return output


class DenseBlock(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation: str = "relu",
    ):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features)
        self.act = get_activation_layer(activation)()

    def forward(self, x):
        return self.act(self.fc(x))


class MISLNet_v4(BaseModel):
    arch = {
        "p256": [
            ("conv1", -1, 96, 7, 2, 1, "valid"),
            ("conv2", 96, 64, 5, 1, 2, "same"),
            ("conv3", 64, 64, 5, 1, 2, "same"),
            ("conv4", 64, 128, 1, 1, 4, "same"),
            ("fc1", 6 * 6 * 128, 200, "gelu"),
            ("fc2", 200, 200, "gelu"),
        ],
        "p256_3fc_256e": [
            ("conv1", -1, 96, 7, 2, 1, "valid"),
            ("conv2", 96, 64, 5, 1, 2, "same"),
            ("conv3", 64, 64, 5, 1, 2, "same"),
            ("conv4", 64, 128, 1, 1, 4, "same"),
            ("fc1", 6 * 6 * 128, 1024, "gelu"),
            ("fc2", 1024, 512, "gelu"),
            ("fc3", 512, 256, "gelu"),
        ],
        "p128": [
            ("conv1", -1, 128, 7, 2, 1, "valid"), # downsizing
            ("conv2", 128, 96, 5, 1, 2, "same"),
            ("conv2", 96, 96, 3, 2, 1, "valid"), # downsizing
            ("conv3", 96, 64, 5, 1, 2, "same"),
            ("conv2", 64, 64, 3, 2, 1, "valid"), # downsizing
            ("conv4", 64, 128, 1, 1, 4, "same"),
            ("conv2", 128, 128, 3, 2, 1, "valid"), # downsizing
            ("fc1", 6 * 6 * 128, 200, "gelu"),
            ("fc2", 200, 200, "gelu"),
        ],
        "p96": [
            ("conv1", -1, 96, 7, 2, "valid"),
            ("conv2", 96, 64, 5, 1, "same"),
            ("conv3", 64, 64, 5, 1, "same"),
            ("conv4", 64, 128, 1, 1, "same"),
            ("fc1", 8 * 4 * 64, 200, "gelu"),
            ("fc2", 200, 200, "gelu"),
        ],
        "p64": [
            ("conv1", -1, 96, 7, 2, "valid"),
            ("conv2", 96, 64, 5, 1, "same"),
            ("conv3", 64, 64, 5, 1, "same"),
            ("conv4", 64, 128, 1, 1, "same"),
            ("fc1", 2 * 4 * 64, 200, "gelu"),
            ("fc2", 200, 200, "gelu"),
        ],
    }

    def __init__(
        self,
        patch_size: int,
        variant: str,
        num_classes: int = 0,
        num_constrained_filters: int = 32,
        num_rgb_filters: int = 32,
        num_heads: int = 8,
        mlp_ratio: float = 4,
        num_blocks: int = 4,
        qkv_bias: bool = True,
        **kwargs,
    ):
        super().__init__(patch_size, num_classes)
        self.variant = variant
        self.chosen_arch = self.arch[variant]
        self.num_constrained_filters = num_constrained_filters
        self.num_rgb_filters = num_rgb_filters

        self.constrained_conv = FusedConstrainedConvWithSelfAttention(
            num_constrained_filters=num_constrained_filters,
            num_rgb_filters=num_rgb_filters,
            input_chan=3,
            patch_size=patch_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_blocks=num_blocks,
            qkv_bias=qkv_bias,
        )

        self.conv_blocks = []
        self.fc_blocks = []
        for block in self.chosen_arch:
            if block[0].startswith("conv"):
                self.conv_blocks.append(
                    FusedMBConv(
                        in_features=(self.constrained_conv.output_chan if block[1] == -1 else block[1]),
                        out_features=block[2],
                        kernel_size=block[3],
                        stride=block[4],
                        expansion_factor=block[5],
                        padding=block[6],
                    )
                )
            elif block[0].startswith("fc"):
                self.fc_blocks.append(
                    DenseBlock(
                        in_features=block[1],
                        out_features=block[2],
                        activation=block[3],
                    )
                )

        self.conv_blocks = nn.Sequential(*self.conv_blocks)
        self.fc_blocks = nn.Sequential(*self.fc_blocks)

        self.register_buffer("flatten_index_permutation", torch.tensor([0, 1, 2, 3], dtype=torch.long))

        if self.num_classes > 0:
            self.output = nn.Linear(self.chosen_arch[-1][2], self.num_classes)

    def forward(self, x):
        x = self.constrained_conv(x)
        x = self.conv_blocks(x)
        x = x.permute(*self.flatten_index_permutation)
        x = x.flatten(1, -1)
        x = self.fc_blocks(x)
        if self.num_classes > 0:
            x = self.output(x)
        return x

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if "flatten_index_permutation" not in state_dict:
            super().load_state_dict(state_dict, False, assign)
        else:
            super().load_state_dict(state_dict, strict, assign)
