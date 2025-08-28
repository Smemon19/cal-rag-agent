from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from eval.rubric import score_item, aggregate
from rag_agent import run_rag_agent
from utils import resolve_collection_name


async def evaluate(collection: str, limit: int | None) -> Dict[str, Any]:
    dataset_path = Path("eval/dataset.jsonl")
    items: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    if limit is not None:
        items = items[: int(limit)]

    rows: List[Dict[str, Any]] = []
    for it in items:
        qid = it.get("qid")
        q = it.get("question")
        try:
            ans = await run_rag_agent(q, collection_name=collection, n_results=5)
        except Exception as e:
            ans = f"[ERROR] {e}"
        row = score_item(ans, it)
        row["answer"] = ans
        rows.append(row)
        print(f"{qid}: acc={row['accuracy']} comp={row['completeness']} grd={row['grounding']} total={row['total']} missing={row['missing']}")

    agg = aggregate(rows)
    return {"scores": rows, "aggregate": agg}


def write_report(result: Dict[str, Any], out_path: str) -> None:
    scores = result["scores"]
    agg = result["aggregate"]
    lines: List[str] = []
    lines.append(f"# Eval Report\n")
    lines.append(f"Average total: {agg['avg_total']:.2f} / 6\n")
    lines.append(f"Pass rate (>=5/6): {agg['pass_rate']*100:.1f}%\n")
    lines.append("\n## Details\n")
    lines.append("| QID | Accuracy | Completeness | Grounding | Total | Missing |\n")
    lines.append("| --- | --- | --- | --- | --- | --- |\n")
    for r in scores:
        lines.append(
            f"| {r['qid']} | {r['accuracy']} | {r['completeness']} | {r['grounding']} | {r['total']} | {', '.join(r['missing'])} |\n"
        )
    Path(out_path).write_text("".join(lines), encoding="utf-8")
    print(f"Wrote report to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run RAG eval and score")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="eval/report.md")
    args = parser.parse_args()

    collection = resolve_collection_name(args.collection)
    print(f"[eval] Using collection: {collection}")

    result = asyncio.run(evaluate(collection, args.limit))
    write_report(result, args.out)


if __name__ == "__main__":
    main()


