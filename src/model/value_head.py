import torch
import torch.nn as nn


# value head: 输入CLS特征 (B,emb_dim)
# 输出：标量预测值(B,1), 可通过tanh约束在[-1,1]
class ValueHead(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, emb_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(emb_dim, 1)

    def forward(self, x):
        h = self.relu(self.linear1(x))
        value = self.linear2(h)  # (B,1)
        # 可选择对输出做tanh:
        value = torch.tanh(value)
        return value
