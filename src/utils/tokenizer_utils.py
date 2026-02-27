"""
Token 长度统计工具。

优先使用 Qwen2.5-Coder tokenizer（与训练模型一致），
回退到 tiktoken cl100k_base，最后回退到字符估算。
"""

from typing import Optional


_ENCODER = None
_ENCODER_TYPE = None  # "qwen" | "tiktoken" | "char"


def get_encoder():
    """延迟加载 tokenizer，按优先级尝试。"""
    global _ENCODER, _ENCODER_TYPE
    if _ENCODER_TYPE is not None:
        return _ENCODER

    # 优先：Qwen2.5-Coder tokenizer（与训练模型一致）
    try:
        from transformers import AutoTokenizer
        _ENCODER = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            trust_remote_code=True,
        )
        _ENCODER_TYPE = "qwen"
        print(f"[tokenizer] Using Qwen2.5-Coder-7B-Instruct tokenizer")
        return _ENCODER
    except Exception:
        pass

    # 回退：tiktoken cl100k_base
    try:
        import tiktoken
        _ENCODER = tiktoken.get_encoding("cl100k_base")
        _ENCODER_TYPE = "tiktoken"
        print(f"[tokenizer] Qwen tokenizer unavailable, using tiktoken cl100k_base")
        return _ENCODER
    except ImportError:
        pass

    # 最终回退：字符估算
    _ENCODER = None
    _ENCODER_TYPE = "char"
    print(f"[tokenizer] No tokenizer available, using char/4 estimation")
    return _ENCODER


def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数量。

    优先级：Qwen tokenizer > tiktoken > 字符数 / 4
    """
    encoder = get_encoder()

    if _ENCODER_TYPE == "qwen":
        return len(encoder.encode(text))
    elif _ENCODER_TYPE == "tiktoken":
        return len(encoder.encode(text))
    else:
        # 粗略估算：1 token ≈ 4 字符
        return len(text) // 4


def estimate_tokens_batch(texts: list) -> list:
    """批量估算 token 数量。"""
    return [estimate_tokens(t) for t in texts]
