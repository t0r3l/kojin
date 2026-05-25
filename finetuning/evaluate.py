"""Evaluate fine-tuned model vs reference model on the eval dataset.

Runs each question from eval.jsonl through both models, executes the generated
SQL against DuckDB, and computes success rates and comparison metrics.

Usage:
    python -m finetuning.evaluate \
        --finetuned-model-id YOUR_CUSTOM_MODEL_ARN \
        [--reference-model-id amazon.nova-micro-v1:0] \
        [--region eu-west-1] \
        [--data-dir ./finetuning/data]

Output:
    - finetuning/data/eval_results.json (detailed per-question results)
    - Console summary with success rates
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kojin_common import CSV_PATH, DUCKDB_PATH, DUCKDB_TABLE


def _ensure_duckdb() -> str:
    """Build DuckDB file from CSV if needed."""
    if not os.path.exists(DUCKDB_PATH):
        if not os.path.exists(CSV_PATH):
            print(f"ERROR: {CSV_PATH} not found. Run data_prep_nutriments.py first.")
            sys.exit(1)
        con = duckdb.connect(DUCKDB_PATH)
        con.execute(
            f"CREATE TABLE {DUCKDB_TABLE} AS "
            f"SELECT * FROM read_csv_auto('{CSV_PATH}', sample_size=-1)"
        )
        con.close()
    return DUCKDB_PATH


def _execute_sql_safe(sql: str, db_path: str) -> tuple[bool, int, str | None]:
    """Execute SQL against DuckDB. Returns (success, row_count, error_msg)."""
    try:
        con = duckdb.connect(db_path, read_only=True)
        result = con.execute(sql).fetchall()
        con.close()
        return True, len(result), None
    except Exception as e:
        return False, 0, str(e)


def _invoke_model(model_id: str, question: str, system_prompt: str, region: str) -> tuple[str, float]:
    """Invoke a Bedrock model, return (generated_sql, latency_ms)."""
    import boto3

    bedrock_rt = boto3.client("bedrock-runtime", region_name=region)

    t0 = time.perf_counter()
    response = bedrock_rt.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": question}]}],
        system=[{"text": system_prompt}],
        inferenceConfig={"maxTokens": 800, "temperature": 0.0},
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    output_msg = response.get("output", {}).get("message", {})
    content_blocks = output_msg.get("content", [])
    raw_text = content_blocks[0].get("text", "") if content_blocks else ""

    # Clean SQL (strip markdown fences)
    import re
    m = re.search(r"```(?:sql)?\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
    sql = m.group(1) if m else raw_text.strip()
    sql = sql.rstrip(";").strip()

    return sql, latency_ms


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned vs reference model")
    parser.add_argument("--finetuned-model-id", required=True,
                        help="ARN or model ID of the fine-tuned Llama on Bedrock")
    parser.add_argument("--reference-model-id", default="amazon.nova-micro-v1:0",
                        help="Reference model ID")
    parser.add_argument("--region", default="eu-west-1", help="AWS region")
    parser.add_argument("--data-dir", default="./finetuning/data", help="Dir with eval.jsonl")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit eval to N samples")
    args = parser.parse_args()

    eval_path = os.path.join(args.data_dir, "eval.jsonl")
    if not os.path.exists(eval_path):
        print(f"ERROR: {eval_path} not found. Run `python -m finetuning.generate_dataset` first.")
        sys.exit(1)

    db_path = _ensure_duckdb()

    # Load eval samples — supports both old format (messages[]+system role)
    # and new bedrock-conversation-2023 format (schemaVersion + system top-level)
    samples = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("schemaVersion") == "bedrock-conversation-2023":
                # New format: system is top-level list, messages only user/assistant
                system_blocks = record.get("system", [])
                system_prompt = system_blocks[0]["text"] if system_blocks else ""
                msgs = record.get("messages", [])
                user_msg = next((m for m in msgs if m["role"] == "user"), {})
                asst_msg = next((m for m in msgs if m["role"] == "assistant"), {})
                content_u = user_msg.get("content", [])
                content_a = asst_msg.get("content", [])
                question = content_u[0]["text"] if content_u else ""
                expected_sql = content_a[0]["text"] if content_a else ""
            else:
                # Legacy format: system/user/assistant all in messages[]
                msgs = record.get("messages", [])
                system_prompt = next((m["content"] for m in msgs if m["role"] == "system"), "")
                question = next((m["content"] for m in msgs if m["role"] == "user"), "")
                expected_sql = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            samples.append({
                "question": question,
                "expected_sql": expected_sql,
                "system_prompt": system_prompt,
            })

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Evaluating {len(samples)} questions against both models…\n")
    print(f"  Fine-tuned : {args.finetuned_model_id}")
    print(f"  Reference  : {args.reference_model_id}")
    print(f"  Region     : {args.region}")
    print()

    results = []
    ft_ok = 0
    ref_ok = 0
    ft_sql_match = 0
    ref_sql_match = 0
    ft_total_ms = 0.0
    ref_total_ms = 0.0

    for i, sample in enumerate(samples):
        q = sample["question"]
        expected = sample["expected_sql"]
        sys_prompt = sample["system_prompt"]

        print(f"  [{i+1}/{len(samples)}] {q[:80]}…" if len(q) > 80 else f"  [{i+1}/{len(samples)}] {q}")

        # Fine-tuned model
        try:
            ft_sql, ft_ms = _invoke_model(args.finetuned_model_id, q, sys_prompt, args.region)
            ft_exec_ok, ft_rows, ft_err = _execute_sql_safe(ft_sql, db_path)
        except Exception as e:
            ft_sql, ft_ms, ft_exec_ok, ft_rows, ft_err = "", 0.0, False, 0, str(e)

        # Reference model
        try:
            ref_sql, ref_ms = _invoke_model(args.reference_model_id, q, sys_prompt, args.region)
            ref_exec_ok, ref_rows, ref_err = _execute_sql_safe(ref_sql, db_path)
        except Exception as e:
            ref_sql, ref_ms, ref_exec_ok, ref_rows, ref_err = "", 0.0, False, 0, str(e)

        ft_total_ms += ft_ms
        ref_total_ms += ref_ms
        if ft_exec_ok:
            ft_ok += 1
        if ref_exec_ok:
            ref_ok += 1
        if ft_sql.strip().lower() == expected.strip().lower():
            ft_sql_match += 1
        if ref_sql.strip().lower() == expected.strip().lower():
            ref_sql_match += 1

        results.append({
            "question": q,
            "expected_sql": expected,
            "finetuned": {
                "sql": ft_sql,
                "exec_ok": ft_exec_ok,
                "rows": ft_rows,
                "error": ft_err,
                "latency_ms": round(ft_ms, 1),
                "exact_match": ft_sql.strip().lower() == expected.strip().lower(),
            },
            "reference": {
                "sql": ref_sql,
                "exec_ok": ref_exec_ok,
                "rows": ref_rows,
                "error": ref_err,
                "latency_ms": round(ref_ms, 1),
                "exact_match": ref_sql.strip().lower() == expected.strip().lower(),
            },
        })

    # Summary
    n = len(samples)
    print("\n" + "═" * 70)
    print("RÉSULTATS D'ÉVALUATION")
    print("═" * 70)
    print(f"\n{'Métrique':<35} {'Fine-tuné':<18} {'Référence':<18}")
    print(f"{'─' * 35} {'─' * 18} {'─' * 18}")
    print(f"{'SQL exécutable (DuckDB OK)':<35} {ft_ok}/{n} ({ft_ok/n*100:.1f}%)     {ref_ok}/{n} ({ref_ok/n*100:.1f}%)")
    print(f"{'Match exact SQL attendu':<35} {ft_sql_match}/{n} ({ft_sql_match/n*100:.1f}%)     {ref_sql_match}/{n} ({ref_sql_match/n*100:.1f}%)")
    print(f"{'Latence moyenne (ms)':<35} {ft_total_ms/n:.0f}ms           {ref_total_ms/n:.0f}ms")
    print()

    # Save detailed results
    output_path = os.path.join(args.data_dir, "eval_results.json")
    summary = {
        "finetuned_model": args.finetuned_model_id,
        "reference_model": args.reference_model_id,
        "n_samples": n,
        "scores": {
            "finetuned": {
                "exec_success_rate": round(ft_ok / n, 4),
                "exact_match_rate": round(ft_sql_match / n, 4),
                "avg_latency_ms": round(ft_total_ms / n, 1),
            },
            "reference": {
                "exec_success_rate": round(ref_ok / n, 4),
                "exact_match_rate": round(ref_sql_match / n, 4),
                "avg_latency_ms": round(ref_total_ms / n, 1),
            },
        },
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Détails sauvegardés → {output_path}")


if __name__ == "__main__":
    main()
