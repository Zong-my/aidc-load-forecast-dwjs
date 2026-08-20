"""通用工具函数: 日志、文件I/O、数据格式辅助。"""

import json
import logging
import os
import sys

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, RESULTS_DIR


def get_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """创建带控制台和可选文件输出的 logger。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def ensure_dir(path: str) -> str:
    """确保目录存在，返回路径。"""
    os.makedirs(path, exist_ok=True)
    return path


def results_path(case: str, *subpath) -> str:
    """构造 results/<case>/... 路径并确保目录存在。"""
    path = os.path.join(RESULTS_DIR, case, *subpath)
    ensure_dir(os.path.dirname(path))
    return path


def load_monitor_data(csv_path: str) -> pd.DataFrame:
    """加载监控 CSV 并确保 elapsed_s 列正确。"""
    df = pd.read_csv(csv_path)
    if "elapsed_s" not in df.columns and "timestamp" in df.columns:
        df["elapsed_s"] = df["timestamp"] - df["timestamp"].iloc[0]
    return df


def save_json(data: dict, filepath: str, indent: int = 2):
    """保存 JSON 文件。"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=indent, cls=NpEncoder, ensure_ascii=False)


def load_json(filepath: str) -> dict:
    """加载 JSON 文件。"""
    with open(filepath) as f:
        return json.load(f)
