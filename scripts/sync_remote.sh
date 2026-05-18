#!/bin/bash
# ============================================================
# 在远端（PolyU VM）运行：整理训练产出 → git push
# 本地：git pull 后 Claude 可以直接分析 remote_logs/polyu/
#
# 用法（在远端 PROJECT_DIR 下）：
#   bash scripts/sync_remote.sh
#
# 产出（committed 到当前分支）：
#   remote_logs/polyu/SUMMARY.md             状态概览 + 最近日志尾部
#   remote_logs/polyu/grpo_polyu.txt         训练日志尾部（截断防过大）
#   remote_logs/polyu/metrics.csv            从 console 解析出的 step 指标
#   remote_logs/polyu/errors_warnings.txt    error/exception/nan/oom 抽样
# ============================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

OUT_DIR="remote_logs/polyu"
LOG_SRC="logs/grpo_polyu.log"
MAX_LINES=120000       # 日志尾部行数上限，约 20-40MB，GitHub 友好

mkdir -p "$OUT_DIR"

TIMESTAMP="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"

# === 1. 日志尾部 ===
if [ -f "$LOG_SRC" ]; then
    tail -n "$MAX_LINES" "$LOG_SRC" > "$OUT_DIR/grpo_polyu.txt"
    LOG_SIZE_MB=$(du -m "$LOG_SRC" | awk '{print $1}')
    LOG_TOTAL_LINES=$(wc -l < "$LOG_SRC")
    LOG_TAIL_LINES=$(wc -l < "$OUT_DIR/grpo_polyu.txt")
else
    echo "[warn] $LOG_SRC not found — 是否未启动训练，或日志路径不同？"
    LOG_SIZE_MB="?"; LOG_TOTAL_LINES="?"; LOG_TAIL_LINES="?"
fi

# === 2. 错误 / 警告抽样 ===
if [ -f "$LOG_SRC" ]; then
    {
        echo "# errors / warnings / NaN / OOM grep (last 500 hits)"
        echo "Source: $LOG_SRC"
        echo ""
        grep -inE "error|traceback|exception|\bwarn\b|\bnan\b|\boom\b|cuda out of memory" \
             "$LOG_SRC" 2>/dev/null | tail -n 500
    } > "$OUT_DIR/errors_warnings.txt"
fi

# === 3. 解析 step 级指标 → CSV ===
python3 - <<'PYEOF'
import re, csv
from pathlib import Path

src = Path("logs/grpo_polyu.log")
out = Path("remote_logs/polyu/metrics.csv")
if not src.exists():
    raise SystemExit(0)

# 兼容 step / global_step，key:value 与 key=value 都接受
step_re = re.compile(r"\b(?:global_step|step)\b[\s:=]+(\d+)\b")
kv_re   = re.compile(r"([A-Za-z][\w./-]*)[\s]*[:=][\s]*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

rows, keys = [], set()
with src.open(errors="ignore") as f:
    for line in f:
        m = step_re.search(line)
        if not m:
            continue
        step = int(m.group(1))
        kvs = {"step": step}
        for k, v in kv_re.findall(line):
            kl = k.lower()
            if kl in ("step", "global_step"):
                continue
            try:
                kvs[k] = float(v)
            except ValueError:
                pass
        if len(kvs) >= 2:
            rows.append(kvs)
            keys.update(kvs.keys())

if rows:
    cols = ["step"] + sorted(keys - {"step"})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[metrics] {len(rows)} rows, {len(cols)-1} metric cols → {out}")
else:
    print("[metrics] 没解析到 step-level 指标（日志格式可能与正则不符）")
PYEOF

# === 4. 状态概览 SUMMARY.md ===
{
    echo "# PolyU 训练同步快照"
    echo ""
    echo "- 时间: $TIMESTAMP"
    echo "- 主机: $HOSTNAME_SHORT"
    echo "- 项目: $PROJECT_DIR"
    echo ""
    echo "## 日志"
    echo "- 源文件: \`$LOG_SRC\`"
    echo "- 大小: ${LOG_SIZE_MB} MB"
    echo "- 总行数: ${LOG_TOTAL_LINES}"
    echo "- 同步尾部行数: ${LOG_TAIL_LINES}（上限 $MAX_LINES）"
    echo ""
    echo "## 正在运行的训练进程"
    echo '```'
    ps -eo pid,etime,pcpu,pmem,cmd 2>/dev/null \
        | grep -E "verl|main_ppo|ray::" | grep -v grep \
        || echo "(没有匹配到 verl/main_ppo/ray 进程)"
    echo '```'
    echo ""
    echo "## GPU 状态"
    echo '```'
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv 2>/dev/null || echo "(nvidia-smi 不可用)"
    echo '```'
    echo ""
    if [ -f "$LOG_SRC" ]; then
        echo "## 日志最后 40 行"
        echo '```'
        tail -n 40 "$LOG_SRC"
        echo '```'
    fi
} > "$OUT_DIR/SUMMARY.md"

# === 5. git commit + push ===
echo ""
echo "[git] staging $OUT_DIR/"
git add "$OUT_DIR" || true

if git diff --cached --quiet; then
    echo "[git] 没有变化，跳过 commit"
else
    BRANCH="$(git rev-parse --abbrev-ref HEAD)"
    if git commit -m "sync(polyu): training snapshot @ $TIMESTAMP" \
                  -m "host=$HOSTNAME_SHORT log_lines=$LOG_TOTAL_LINES"; then
        echo "[git] pushing to origin/$BRANCH ..."
        git push origin "$BRANCH" || \
            echo "[git] push 失败，检查认证/分支保护"
    else
        echo "[git] commit 失败（可能 user.name/user.email 未配置）"
        echo "      请先：git config user.email 'you@example.com' && git config user.name 'You'"
    fi
fi

echo ""
echo "[done] 产出位于：$OUT_DIR/"
