import yaml
import os


def load_config(config_path):
    """
    加载 YAML 配置文件并返回 Python 字典。

    参数:
        config_path (str): 配置文件的路径。

    返回:
        dict: 配置文件内容作为字典。

    异常:
        FileNotFoundError: 如果配置文件不存在。
        yaml.YAMLError: 如果文件格式不正确。
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return config
    except yaml.YAMLError as e:
        raise ValueError(f"解析配置文件时发生错误: {e}")
