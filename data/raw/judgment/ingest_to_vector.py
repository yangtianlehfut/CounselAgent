#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
判决书摘要向量入库（chromadb 1.5.9）

- 输入：单个 csv，或目录（递归扫描所有 .csv）
- 向量化对象：abstract_text；记录 id 用 case_id，空案号报错跳过
- 写入：Chroma PersistentClient 本地持久化，余弦距离，按 id upsert，可重复执行
- 嵌入配置全部来自 src/config.py（目前只支持 ollama）

用法：
    python ingest_to_vector.py --input ./judgment_text_10 --collection judgments
    python ingest_to_vector.py --input ./judgment_text_10 --per-cause
    python ingest_to_vector.py --input ./judgment_text_10 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import config  # noqa: E402


def iter_csv_files(input_path: str | Path) -> List[Path]:
    """文件则返回自身；目录则递归扫描全部 .csv（跳过隐藏文件与 Office 临时文件）。"""
    p = Path(input_path).expanduser()
    if not p.exists():
        sys.exit(f"输入路径不存在：{p}")
    if p.is_file():
        if p.suffix.lower() != ".csv":
            sys.exit(f"不是 csv 文件：{p}")
        return [p]
    files = [
        f for f in sorted(p.rglob("*.csv"))
        if not any(part.startswith(".") for part in f.relative_to(p).parts)
        and not f.name.startswith("~$")
    ]
    if not files:
        sys.exit(f"目录下没有找到 csv 文件：{p}")
    return files


def parse_tags(raw: str) -> List[str]:
    """'#甲#乙#丙' -> ['甲', '乙', '丙']"""
    return [t.strip() for t in (raw or "").split("#") if t.strip()]


@dataclass
class Stats:
    files: int = 0
    rows: int = 0
    valid: int = 0
    skipped_empty: int = 0
    skipped_no_case_id: int = 0
    skipped_existing: int = 0
    written: int = 0
    problems: List[str] = field(default_factory=list)


def load_cases(csv_path: Path, stats: Stats, strict: bool = False) -> List[Dict[str, Any]]:
    """读取单个 csv，返回 [{id, document, metadata}]。"""
    records: List[Dict[str, Any]] = []
    stats.files += 1

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            stats.problems.append(f"{csv_path.name}: 空文件或无表头")
            return records
        missing = {"abstract_text", "cause", "case_id", "case_name", "tags", "judgment_result"} - set(reader.fieldnames)
        if missing:
            msg = f"{csv_path.name}: 缺少列 {sorted(missing)}"
            if strict:
                sys.exit(msg)
            stats.problems.append(msg + "，已跳过该文件")
            return records

        for lineno, row in enumerate(reader, start=2):
            stats.rows += 1
            abstract = (row.get("abstract_text") or "").strip()
            if not abstract:
                stats.skipped_empty += 1
                print(f"[skip] {csv_path.name}:{lineno} case_id={row.get('case_id')} 无 abstract_text")
                continue

            metadata = {
                "cause": (row.get("cause") or "").strip(),
                "case_name": (row.get("case_name") or "").strip(),
                "judgment_result": (row.get("judgment_result") or "").strip(),
                "tags": parse_tags(row.get("tags") or ""),
            }

            case_id = (row.get("case_id") or "").strip()
            if not case_id:
                stats.skipped_no_case_id += 1
                print(f"[error] {csv_path.name}:{lineno} case_id 为空，已跳过该行")
                stats.problems.append(f"{csv_path.name}:{lineno} case_id 为空，未入库")
                continue

            records.append({"id": case_id, "document": abstract, "metadata": metadata})

    return records


def sanity_check(records: List[Dict[str, Any]], stats: Stats) -> None:
    tag_re = re.compile(r"^[\u4e00-\u9fa5A-Za-z0-9]{2,6}$")
    seen: Dict[str, int] = {}
    for r in records:
        m = r["metadata"]
        seen[r["id"]] = seen.get(r["id"], 0) + 1
        for field_name in ("cause", "case_name", "judgment_result"):
            if not m[field_name]:
                stats.problems.append(f"{r['id']}: {field_name} 为空")
        if not m["tags"]:
            stats.problems.append(f"{r['id']}: tags 为空")
        for t in m["tags"]:
            if not tag_re.match(t):
                stats.problems.append(f"{r['id']}: 标签不规范 -> {t}")
        if not 250 <= len(r["document"]) <= 400:
            stats.problems.append(f"{r['id']}: 摘要字数 {len(r['document'])}（建议 250-400）")
    for k, v in seen.items():
        if v > 1:
            stats.problems.append(f"id 重复 {v} 次：{k}（upsert 会互相覆盖）")

    if stats.problems:
        print(f"[check] 发现 {len(stats.problems)} 条提示：")
        for p in stats.problems:
            print("  -", p)
    else:
        print("[check] 校验通过：id 唯一、元数据完整（含 judgment_result）、标签格式合规、摘要字数达标")


def build_embedding_function(model_name: str) -> "OllamaEmbeddingFunction":
    """本地 Ollama 嵌入函数（地址、超时用 ollama 默认配置）；同一个实例既用于写入，也挂到 collection 上供查询时复用。"""
    if config.EMBEDDING_PROVIDER != "ollama":
        sys.exit(f"暂不支持的 EMBEDDING_PROVIDER：{config.EMBEDDING_PROVIDER}（目前只支持 ollama）")

    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

    return OllamaEmbeddingFunction(model_name=model_name)


def embed_texts(ef, texts: Iterable[str], batch_size: int = 16, retries: int = 3) -> List[List[float]]:
    """按批次调用本地 Ollama 嵌入；失败时退避重试。"""
    texts = list(texts)
    vectors: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        for attempt in range(1, retries + 1):
            try:
                vectors.extend(ef(batch))
                print(f"[embed] {min(start + batch_size, len(texts))}/{len(texts)}")
                break
            except Exception as exc:
                if attempt == retries:
                    raise
                wait = 2 * attempt
                print(f"[embed] 第 {attempt} 次失败：{exc}，{wait}s 后重试")
                time.sleep(wait)
    return vectors


def create_collection(client, name: str, ef):
    """chroma 1.x 用 configuration 指定距离；绑定 embedding_function，查询时自动用同一模型。"""
    kwargs: Dict[str, Any] = {"name": name, "embedding_function": ef}
    try:
        return client.get_or_create_collection(
            **kwargs, configuration={"hnsw": {"space": "cosine"}}
        )
    except Exception:
        return client.get_or_create_collection(**kwargs, metadata={"hnsw:space": "cosine"})


def to_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "cause": meta["cause"],
        "case_name": meta["case_name"],
        "judgment_result": meta["judgment_result"],
    }
    out["tags"] = list(meta["tags"])
    return out


def upsert(
    records: List[Dict[str, Any]],
    persist_dir: Path,
    collection: str,
    embed_batch: int,
    upsert_batch: int,
    ef,
    skip_existing: bool = False,
) -> Tuple[int, int]:
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    col = create_collection(client, collection, ef)
    before = col.count()

    print("[meta] tags 以字符串列表存储")

    if skip_existing:
        existing = col.get(ids=[r["id"] for r in records]).get("ids", [])
        existing_ids = set(existing)
        skipped = [r for r in records if r["id"] in existing_ids]
        if skipped:
            print(f"[skip] 已存在 {len(skipped)} 条，跳过 embedding")
        records = [r for r in records if r["id"] not in existing_ids]
        if not records:
            print(f"[done] collection={collection}：无可新增数据，当前 {col.count()} 条")
            return 0, len(skipped)
    else:
        skipped = []

    total = len(records)
    for start in range(0, total, upsert_batch):
        chunk = records[start:start + upsert_batch]
        vectors = embed_texts(ef, [c["document"] for c in chunk], batch_size=embed_batch)
        col.upsert(
            ids=[c["id"] for c in chunk],
            documents=[c["document"] for c in chunk],
            metadatas=[to_meta(c["metadata"]) for c in chunk],
            embeddings=vectors,
        )
        print(f"[store] {min(start + upsert_batch, total)}/{total}")

    print(f"[done] collection={collection}：写入前 {before} 条，写入后 {col.count()} 条")
    return col.count() - before, len(skipped)


def main() -> None:
    ap = argparse.ArgumentParser(description="abstract_text 向量入库（chromadb 1.5.9）")
    ap.add_argument("--input", required=True, help="csv 文件路径，或包含多个 csv 的目录（递归扫描）")
    ap.add_argument("--persist-dir", default=str(config.VECTOR_DB_PATH), help="Chroma 持久化目录（默认取 config.VECTOR_DB_PATH）")
    ap.add_argument("--collection", default="judgments", help="collection 名称（--per-cause 时忽略）")
    ap.add_argument("--per-cause", action="store_true", help="按 cause 分库，每个案由一个 collection")
    ap.add_argument("--embed-model", default=config.EMBEDDING_MODEL_NAME, help="嵌入模型（默认取 config.EMBEDDING_MODEL_NAME）")
    ap.add_argument("--embed-batch", type=int, default=config.EMBEDDING_BATCH_SIZE, help="embedding 批次（默认取 config.EMBEDDING_BATCH_SIZE）")
    ap.add_argument("--upsert-batch", type=int, default=100, help="写库批次")
    ap.add_argument("--skip-existing", action="store_true", help="已存在的 id 不再重复向量化（省 embedding 费用）")
    ap.add_argument("--dry-run", action="store_true", help="只解析与校验，不调用 embedding、不写库")
    args = ap.parse_args()

    files = iter_csv_files(args.input)
    print(f"[scan] 共 {len(files)} 个 csv 文件")
    for f in files:
        print("  -", f)

    stats = Stats()
    records: List[Dict[str, Any]] = []
    for f in files:
        records.extend(load_cases(f, stats, strict=False))

    stats.valid = len(records)
    sanity_check(records, stats)
    print(
        f"[load] 文件 {stats.files} 个，数据行 {stats.rows}，有效 {stats.valid}，"
        f"空摘要跳过 {stats.skipped_empty}，空案号跳过 {stats.skipped_no_case_id}"
    )

    if not records:
        sys.exit("没有可入库的数据")
    if args.dry_run:
        print("\n[dry-run] 样例：")
        print(records[0])
        return

    ef = build_embedding_function(args.embed_model)
    print(f"[embed] 使用 {config.EMBEDDING_PROVIDER} / {args.embed_model}")

    persist_dir = Path(args.persist_dir).expanduser()
    if args.per_cause:
        by_cause: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            by_cause.setdefault(r["metadata"]["cause"] or "未分类", []).append(r)
        for cause, group in by_cause.items():
            name = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5_-]", "_", cause)
            written, skipped = upsert(
                group, persist_dir, name, args.embed_batch, args.upsert_batch, ef, args.skip_existing
            )
            print(f"[summary] {cause}: 新增/更新 {written} 条，跳过已存在 {skipped} 条\n")
    else:
        written, skipped = upsert(
            records, persist_dir, args.collection, args.embed_batch, args.upsert_batch, ef, args.skip_existing
        )
        print(f"\n[summary] 共处理 {stats.valid} 条，新增/更新 {written} 条，跳过已存在 {skipped} 条")


if __name__ == "__main__":
    main()
