"""
Token 长度统计工具。

使用 tiktoken 快速估算 token 数量（无需加载完整 HF tokenizer）。
"""

from typing import Optional


_ENCODER = None


def get_encoder():
    """延迟加载 tiktoken encoder。"""
    global _ENCODER
    if _ENCODER is None:
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            _ENCODER = None
    return _ENCODER


def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数量。

    优先使用 tiktoken（快速），回退到字符数 / 4 的粗略估算。
    """
    encoder = get_encoder()
    if encoder is not None:
        return len(encoder.encode(text))
    # 粗略估算：1 token ≈ 4 字符
    return len(text) // 4


def estimate_tokens_batch(texts: list) -> list:
    """批量估算 token 数量。"""
    return [estimate_tokens(t) for t in texts]
