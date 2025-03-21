import torch.nn as nn


# policy head: 输入是CLS特征 (B,emb_dim)
# 输出：81维策略分布（未softmax）
class PolicyHead(nn.Module):
    def __init__(self, emb_dim=64, num_actions=81):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        # x: (B,emb_dim)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
