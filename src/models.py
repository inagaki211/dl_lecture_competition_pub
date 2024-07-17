import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# ベースモデル
class BasicConvClassifier(nn.Module):
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

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.5, # ドロップアウト率の調整 default: 0.1
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

# Convモデル
class ConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 256,  # 次元数を増やす
        num_layers: int = 4,  # より層を深く
        kernel_size: int = 3,
        p_drop: float = 0.5
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim, kernel_size, p_drop),
            *[ConvBlock(hid_dim, hid_dim, kernel_size, p_drop) for _ in range(num_layers - 1)]  # より層を深く
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        return self.head(X)

# EEGNetモデル
class EEGNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int = 271,  # 入力チャネル数を271に更新
        hid_dim: int = 64,
        kernel_size: int = 3,
        p_drop: float = 0.5
    ) -> None:
        super(EEGNet, self).__init__()

        self.blocks = nn.Sequential(
            # 第一層の畳み込み層
            nn.Conv2d(1, hid_dim, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2)),  # パディングを修正
            nn.BatchNorm2d(hid_dim),
            nn.ELU(),
            # 第二層の畳み込み層
            nn.Conv2d(hid_dim, hid_dim, kernel_size=(in_channels, 1), groups=hid_dim, padding=(0, 0)),  # 深層畳み込み
            nn.BatchNorm2d(hid_dim),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p_drop),
        )

        self.head = nn.Sequential(
            nn.Linear(hid_dim * (seq_len // 4), num_classes),  # 全結合層の入力サイズを修正
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.unsqueeze(1)  # (batch, channels, time) -> (batch, 1, channels, time)
        X = self.blocks(X)
        X = X.view(X.size(0), -1)  # フラット化
        return self.head(X)


# ResNetモデル
class ResNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        num_blocks: int = 2,
        kernel_size: int = 3,
        p_drop: float = 0.5
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim, kernel_size, p_drop),
            *[ResidualBlock(hid_dim, hid_dim, kernel_size, p_drop) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        return self.head(X)

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        p_drop: float = 0.5
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")

        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X_res = X
        else:
            X_res = self.conv0(X)

        X = self.conv0(X)
        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X)
        X = F.gelu(self.batchnorm1(X))

        X = self.dropout(X)
        return X + X_res

# Transformerモデル
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=num_heads,
            dim_feedforward=hid_dim * 4,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        X = X.permute(2, 0, 1)  # (b, c, t) -> (t, b, c) for transformer
        X = self.transformer_encoder(X)
        X = X.permute(1, 2, 0)  # (t, b, c) -> (b, c, t) back to original
        return self.head(X)

# 被験者情報を扱うモデル
class SubjectEmbedding(nn.Module):
    def __init__(self, num_subjects, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_subjects, embedding_dim)
    
    def forward(self, subject_ids):
        return self.embedding(subject_ids)

class SubjectAwareTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_subjects: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        subject_embedding_dim: int = 32
    ) -> None:
        super().__init__()

        self.subject_embedding = SubjectEmbedding(num_subjects, subject_embedding_dim)

        self.blocks = nn.Sequential(
            nn.Conv1d(in_channels, hid_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim + subject_embedding_dim,
            nhead=num_heads,
            dim_feedforward=(hid_dim + subject_embedding_dim) * 4,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim + subject_embedding_dim, num_classes),
        )

    def forward(self, X: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        subject_emb = self.subject_embedding(subject_ids)
        subject_emb = subject_emb.unsqueeze(2).expand(-1, -1, X.size(2))  # (b, emb_dim, t)

        X = self.blocks(X)
        X = torch.cat([X, subject_emb], dim=1)  # MEGデータと被験者埋め込みを結合

        X = X.permute(2, 0, 1)  # (b, c, t) -> (t, b, c) for transformer
        X = self.transformer_encoder(X)
        X = X.permute(1, 2, 0)  # (t, b, c) -> (b, c, t) back to original

        return self.head(X)