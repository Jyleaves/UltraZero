import torch
import torch.nn as nn
import math


# ViT参数参考:
# - embed_dim = 64
# - num_heads = 4
# - num_layers = 4
# - mlp_dim = 128
# - patch_size = 3x3
# 输入 shape: (B, 5, 9, 9)
# 将9x9分为3x3个patch，每个patch为3x3，共9个patch + 1个CLS token
# 最终得到 (B, 10, embed_dim) 的序列(包括CLS)
# 然后经过TransformerEncoder
# 最终输出CLS token对应的embedding作为全局特征向量

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=5, patch_size=3, emb_dim=64):
        super().__init__()
        # 将每个3x3的patch展开为长度为patch_size*patch_size*in_channels的向量
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.proj = nn.Linear(in_channels * patch_size * patch_size, emb_dim)

    def forward(self, x):
        # x: (B,5,9,9)
        B, C, H, W = x.shape
        # 分patch:共9个patch(3x3大格)
        # 按3x3切片
        patches = []
        # 以3为步长分割9x9为3x3个patch
        for i in range(0, H, self.patch_size):
            for j in range(0, W, self.patch_size):
                patch = x[:, :, i:i + self.patch_size, j:j + self.patch_size]
                # patch: (B,C,3,3)
                patch_flat = patch.reshape(B, -1)  # (B,C*3*3)
                patches.append(patch_flat)
        patches = torch.stack(patches, dim=1)  # (B,9,C*3*3)

        # 线性映射到emb_dim
        return self.proj(patches)  # (B,9,emb_dim)


class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim=64, num_heads=4, mlp_dim=128, num_layers=4, dropout=0.1):
        super().__init__()
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(TransformerBlock(emb_dim, num_heads, mlp_dim, dropout))
        self.layers = nn.Sequential(*encoder_layers)

    def forward(self, x):
        # x: (B,N,emb_dim)
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-Attention
        h = self.norm1(x)
        h_attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + h_attn
        # MLP
        h2 = self.norm2(x)
        h2 = self.mlp(h2)
        x = x + h2
        return x


class ViT(nn.Module):
    def __init__(self, in_channels=5, patch_size=3, emb_dim=64, num_heads=4, num_layers=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 10, emb_dim) * 0.02)  # 9 patches + 1 cls
        self.encoder = TransformerEncoder(emb_dim, num_heads, mlp_dim, num_layers, dropout)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        # x: (B,5,9,9)
        B = x.size(0)
        x = self.patch_embed(x)  # (B,9,emb_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,emb_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B,10,emb_dim)

        x = x + self.pos_embed  # 位置编码
        x = self.encoder(x)
        x = self.norm(x)
        # 取cls token作为全局特征
        cls_repr = x[:, 0, :]  # (B,emb_dim)
        return cls_repr
