import numpy as np
import os


class ReplayBuffer:
    def __init__(self, max_size=500000):
        # max_size: 缓冲区最大容量
        # 数据结构：
        # states: (max_size, 5,9,9)
        # policies: (max_size,81)
        # values: (max_size)
        self.max_size = max_size
        self.states = np.zeros((max_size, 5, 9, 9), dtype=np.float32)
        self.policies = np.zeros((max_size, 81), dtype=np.float32)
        self.values = np.zeros((max_size,), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(self, state, policy, value):
        # state: np.array(5,9,9), policy: np.array(81), value: float
        self.states[self.ptr] = state
        self.policies[self.ptr] = policy
        self.values[self.ptr] = value
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def save(self, file_path):
        """
        保存缓冲区数据到文件。
        :param file_path: 文件路径（例如：'data/replay_buffer/buffer_cycle_0.npz'）
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # 保存数据
        np.savez(
            file_path,
            states=self.states[:self.size],
            policies=self.policies[:self.size],
            values=self.values[:self.size],
            ptr=self.ptr,
            size=self.size
        )

    def load(self, file_path):
        """
        从文件加载缓冲区数据。
        :param file_path: 文件路径（例如：'data/replay_buffer/buffer_cycle_0.npz'）
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Buffer file {file_path} does not exist.")

        # 加载数据
        data = np.load(file_path)
        self.states = np.zeros((self.max_size, 5, 9, 9), dtype=np.float32)
        self.policies = np.zeros((self.max_size, 81), dtype=np.float32)
        self.values = np.zeros((self.max_size,), dtype=np.float32)

        loaded_size = data['states'].shape[0]
        self.states[:loaded_size] = data['states']
        self.policies[:loaded_size] = data['policies']
        self.values[:loaded_size] = data['values']
        self.ptr = data['ptr']
        self.size = data['size']

        print(f"ReplayBuffer loaded from {file_path}")

    def load_and_merge(self, file_path):
        """
        从 file_path 加载对局数据，并合并到当前 ReplayBuffer。
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Buffer file {file_path} does not exist.")

        # 创建一个临时 buffer 来加载文件
        temp_buffer = ReplayBuffer(max_size=self.max_size)
        temp_buffer.load(file_path)

        # 逐条将其合并到当前 buffer
        for i in range(temp_buffer.size):
            state = temp_buffer.states[i]
            policy = temp_buffer.policies[i]
            value = temp_buffer.values[i]
            self.add(state, policy, value)

        print(f"Merged {temp_buffer.size} entries from {file_path} into the current ReplayBuffer.")

    def sample(self, batch_size):
        # 均匀随机采样
        idxs = np.random.randint(0, self.size, size=batch_size)
        return self.states[idxs], self.policies[idxs], self.values[idxs]

    def get_data(self):
        """
        获取缓冲区中所有有效的数据。
        :return: 返回一个元组 (states, policies, values)，其中每个元素是一个 NumPy 数组。
        """
        return self.states[:self.size], self.policies[:self.size], self.values[:self.size]

    def __len__(self):
        return self.size
