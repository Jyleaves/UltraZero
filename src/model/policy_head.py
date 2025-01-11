import torch
import torch.nn as nn


# policy head: 输入是CLS特征 (B,emb_dim)
# 输出：81维策略分布（未softmax）
class PolicyHead(nn.Module):
    def __init__(self, emb_dim=64, num_actions=81):
        super().__init__()
        self.linear = nn.Linear(emb_dim, num_actions)
        # 不加softmax，在外部计算CrossEntropyLoss时会自动从logits中做softmax

    def forward(self, x):
        # x: (B,emb_dim)
        logits = self.linear(x)  # (B,81)
        return logits
