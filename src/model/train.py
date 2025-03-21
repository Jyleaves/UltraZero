import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import time
from tqdm import tqdm  # 导入tqdm库，用于显示进度条

from .vit_model import ViT  # 假设ViT模型定义在vit_model模块中
from .policy_head import PolicyHead  # 假设策略头定义在policy_head模块中
from .value_head import ValueHead  # 假设价值头定义在value_head模块中


# 自定义数据集类，用于加载训练数据
class UltraZeroDataset(Dataset):
    def __init__(self, data):
        """
        初始化数据集
        :param data: 数据元组 (states, policies, values)
                     states: np.array(N, 5, 9, 9)
                     policies: np.array(N, 81)
                     values: np.array(N,)
        """
        self.states, self.policies, self.values = data

    def __len__(self):
        """返回数据集的大小"""
        return len(self.states)

    def __getitem__(self, idx):
        """
        根据索引获取单个数据样本
        :param idx: 数据索引
        :return: state (torch.Tensor), pi_target (torch.Tensor), z (torch.Tensor)
        """
        state = torch.from_numpy(self.states[idx]).float()  # 将state转换为float32类型的张量
        pi = torch.from_numpy(self.policies[idx]).float()  # 将pi_target转换为float32类型的张量
        z = torch.tensor(self.values[idx], dtype=torch.float32)  # 将z转换为float32类型的标量张量
        return state, pi, z


# 定义UltraZero模型，包含ViT、策略头和价值头
class UltraZeroModel(nn.Module):
    def __init__(self, emb_dim=64, num_heads=4, mlp_dim=128, num_layers=4, dropout=0.1):
        """
        初始化模型
        :param emb_dim: ViT的嵌入维度
        :param num_heads: ViT的多头注意力头数
        :param mlp_dim: ViT的MLP层维度
        :param num_layers: ViT的层数
        :param dropout: Dropout概率
        """
        super().__init__()
        self.vit = ViT(in_channels=5, patch_size=3, emb_dim=emb_dim, num_heads=num_heads,
                       mlp_dim=mlp_dim, num_layers=num_layers, dropout=dropout)  # ViT模型
        self.policy_head = PolicyHead(emb_dim, 81)  # 策略头，输出81维的概率分布
        self.value_head = ValueHead(emb_dim)  # 价值头，输出标量值

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，形状为 (B, 5, 9, 9)
        :return: policy_logits (策略输出), value (价值输出)
        """
        cls_repr = self.vit(x)  # 通过ViT提取特征
        policy_logits = self.policy_head(cls_repr)  # 通过策略头生成策略输出
        value = self.value_head(cls_repr)  # 通过价值头生成价值输出
        return policy_logits, value


def train_one_epoch(model, dataloader, optimizer, scaler, device, value_loss_weight=1.0):
    """
    训练一个epoch
    :param model: 训练的模型
    :param dataloader: 数据加载器
    :param optimizer: 优化器
    :param scaler: 混合精度训练的GradScaler
    :param device: 设备（CPU或GPU）
    :param value_loss_weight: 价值损失的权重
    :return: 平均总损失, 平均策略损失, 平均价值损失
    """
    model.train()  # 设置模型为训练模式
    total_loss = 0.0  # 累计总损失
    total_policy_loss = 0.0  # 累计策略损失
    total_value_loss = 0.0  # 累计价值损失

    # 使用tqdm包装dataloader，显示训练进度条
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for state, pi_target, z in progress_bar:
        # 将数据移动到指定设备
        state = state.to(device)
        pi_target = pi_target.to(device)
        z = z.to(device).unsqueeze(-1)  # 将z从标量扩展为 (B, 1) 的形状

        optimizer.zero_grad()  # 清空梯度

        # 混合精度前向传播
        with torch.amp.autocast('cuda'):
            policy_logits, value_pred = model(state)  # 前向传播，获取策略输出和价值输出

            # 计算策略损失：交叉熵损失
            log_probs = F.log_softmax(policy_logits, dim=-1)  # 对策略输出进行log_softmax
            policy_loss = -(pi_target * log_probs).sum(dim=-1).mean()  # 计算负对数似然损失

            # 计算价值损失：均方误差损失
            value_loss = F.mse_loss(value_pred, z)

            # 计算总损失
            loss = policy_loss + value_loss_weight * value_loss

        # 使用混合精度的 scaler 进行反向传播和优化
        scaler.scale(loss).backward()  # 缩放 loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        scaler.step(optimizer)  # 更新模型参数
        scaler.update()  # 更新Scaler

        # 累计损失
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

        # 更新进度条的描述信息，显示当前的平均损失
        progress_bar.set_postfix({
            'loss': total_loss / (progress_bar.n + 1),
            'policy_loss': total_policy_loss / (progress_bar.n + 1),
            'value_loss': total_value_loss / (progress_bar.n + 1)
        })

    n = len(dataloader)  # 获取batch的数量
    return total_loss / n, total_policy_loss / n, total_value_loss / n  # 返回平均损失


def train_model(model, dataset, gpu_ids, batch_size=256, learning_rate=1e-3, epochs=10, weight_decay=0.0001, value_loss_weight=1.0):
    """
    训练模型
    :param gpu_ids: 可用gpu编号
    :param weight_decay: 权重衰减（防止过拟合）
    :param model: 要训练的模型
    :param dataset: 训练数据集
    :param batch_size: 批量大小
    :param learning_rate: 学习率
    :param epochs: 训练的总epoch数
    :param value_loss_weight: 价值损失的权重
    """
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 使用AdamW优化器

    # 初始化 GradScaler
    scaler = torch.amp.GradScaler()

    for epoch in range(epochs):
        start_time = time.time()  # 记录当前epoch的开始时间
        # 训练一个epoch
        loss, pol_loss, val_loss = train_one_epoch(model, dataloader, optimizer, scaler, device, value_loss_weight)
        elapsed = time.time() - start_time  # 计算当前epoch的耗时

        # 使用tqdm.write打印每个epoch的信息，避免与进度条冲突
        tqdm.write(f"\nEpoch {epoch + 1}/{epochs}: loss={loss:.4f}, policy_loss={pol_loss:.4f}, value_loss={val_loss:.4f}, time={elapsed:.2f}s\n")
