"""
代码去重工具。

提供两种去重方式：
1. 精确去重：基于标准化代码 hash
2. 模糊去重：基于 MinHash LSH（需要 datasketch）
"""

import hashlib
import re
from typing import List, Optional, Set, Tuple


def normalize_code(code: str) -> str:
    """标准化代码：去除注释、多余空白、统一换行。"""
    # 去除单行注释
    code = re.sub(r"#[^\n]*", "", code)
    # 去除多行字符串注释（简单版，不处理嵌套）
    code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", "", code, flags=re.DOTALL)
    # 统一空白
    code = re.sub(r"[ \t]+", " ", code)
    # 去除空行
    lines = [line.strip() for line in code.split("\n") if line.strip()]
    return "\n".join(lines)


def code_hash(code: str) -> str:
    """计算标准化代码的 hash。"""
    normalized = normalize_code(code)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def exact_dedup(
    codes: List[str],
    priorities: Optional[List[int]] = None,
) -> List[int]:
    """
    精确去重：基于标准化代码 hash。

    Args:
        codes: 代码列表
        priorities: 优先级列表（值越大优先保留，如 GitHub stars）

    Returns:
        保留的索引列表
    """
    if priorities is None:
        priorities = list(range(len(codes)))

    hash_to_best: dict = {}  # hash -> (index, priority)

    for i, code in enumerate(codes):
        h = code_hash(code)
        if h not in hash_to_best or priorities[i] > hash_to_best[h][1]:
            hash_to_best[h] = (i, priorities[i])

    kept_indices = sorted([idx for idx, _ in hash_to_best.values()])
    return kept_indices


def minhash_dedup(
    codes: List[str],
    threshold: float = 0.85,
    num_perm: int = 128,
    priorities: Optional[List[int]] = None,
) -> List[int]:
    """
    模糊去重：基于 MinHash LSH。

    Args:
        codes: 代码列表
        threshold: Jaccard 相似度阈值
        num_perm: MinHash 排列数
        priorities: 优先级列表

    Returns:
        保留的索引列表
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("Warning: datasketch not installed, falling back to exact dedup")
        return exact_dedup(codes, priorities)

    if priorities is None:
        priorities = list(range(len(codes)))

    # 构建 MinHash
    minhashes = []
    for code in codes:
        m = MinHash(num_perm=num_perm)
        normalized = normalize_code(code)
        # 用 3-gram shingles
        for i in range(max(1, len(normalized) - 2)):
            m.update(normalized[i : i + 3].encode("utf-8"))
        minhashes.append(m)

    # 构建 LSH 索引
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, m in enumerate(minhashes):
        try:
            lsh.insert(str(i), m)
        except ValueError:
            pass  # 重复的 key，跳过

    # 找到重复组，保留优先级最高的
    removed: Set[int] = set()
    for i in range(len(codes)):
        if i in removed:
            continue
        result = lsh.query(minhashes[i])
        duplicates = [int(r) for r in result if int(r) != i and int(r) not in removed]
        for dup_idx in duplicates:
            if priorities[dup_idx] <= priorities[i]:
                removed.add(dup_idx)
            else:
                removed.add(i)
                break

    kept_indices = sorted([i for i in range(len(codes)) if i not in removed])
    return kept_indices
