import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# Self-Attentionプロック
class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).permute(0, 2, 1, 3), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class BasicConvClassifier(nn.Module):
    
#     ベースモデル
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
             X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)
    
        
#     def __init__(
#         self,
#         num_classes: int,
#         seq_len: int,
#         in_channels: int,
#         hid_dim: int = 128,
#         heads: int = 8,
#         attn_dropout: float = 0.1
#     ) -> None:
#         super().__init__()

#         self.blocks = nn.Sequential(
#             ConvBlock(in_channels, hid_dim),
#             ConvBlock(hid_dim, hid_dim),
#             ConvBlock(hid_dim, hid_dim),  # 追加のConvBlock
#         )

#         self.attention = SelfAttention(hid_dim, heads, attn_dropout)

#         self.head = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             Rearrange("b d 1 -> b d"),
#             nn.Linear(hid_dim, num_classes),
#         )

#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         X = self.blocks(X)
#         X = X.permute(0, 2, 1)  # (b, c, t) -> (b, t, c) for attention
#         X = self.attention(X)
#         X = X.permute(0, 2, 1)  # (b, t, c) -> (b, c, t) back to original
#         return self.head(X)


class ConvBlock(nn.Module):
    #ベースライン
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)

#     def __init__(
#         self,
#         in_dim,
#         out_dim,
#         kernel_size: int = 3,
#         p_drop: float = 0.2,  # ドロップアウト率の調整
#     ) -> None:
#         super().__init__()

#         self.in_dim = in_dim
#         self.out_dim = out_dim

#         self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
#         self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
#         self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")  # 追加のConvレイヤー

#         self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
#         self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
#         self.batchnorm2 = nn.BatchNorm1d(num_features=out_dim)  # 追加のBatchNorm

#         self.dropout = nn.Dropout(p_drop)

#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         if self.in_dim == self.out_dim:
#             X = self.conv0(X) + X  # skip connection
#         else:
#             X = self.conv0(X)

#         X = F.gelu(self.batchnorm0(X))

#         X = self.conv1(X) + X  # skip connection
#         X = F.gelu(self.batchnorm1(X))

#         X = self.conv2(X) + X  # 追加のConvレイヤーとskip connection
#         X = F.gelu(self.batchnorm2(X))

#         return self.dropout(X)