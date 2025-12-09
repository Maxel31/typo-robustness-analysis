"""ロギングユーティリティモジュール."""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "nlp2026",
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """ロガーをセットアップする.

    Args:
        name: ロガー名
        level: ログレベル
        log_file: ログファイルのパス（指定した場合はファイルにも出力）

    Returns:
        設定済みのロガー
    """
    logger = logging.getLogger(name)

    # 既に設定済みの場合は再設定しない
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # フォーマッター
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # コンソールハンドラー
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ファイルハンドラー（指定された場合）
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# デフォルトロガー
logger = setup_logger()
