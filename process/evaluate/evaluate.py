import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


def add_period_if_needed(text: str) -> str:
    text = str(text or "").strip()
    if text and not text.endswith("?") and not text.endswith("!"):
        if not text.endswith("."):
            text += "."
    return text


def normalize_think_tags(text: Any) -> str:
    """
    轻量兼容：把 <thinking>...</thinking> 规范成 <think>...</think>。
    主提取逻辑仍然使用用户给的 split('</think>') 规则。
    """
    text = str(text or "")
    text = re.sub(r"<\s*thinking\s*>", "<think>", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/\s*thinking\s*>", "</think>", text, flags=re.IGNORECASE)
    return text


def extract_gt_user_rule(rec: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
    """
    完全对齐用户 get_evaluate() 里的 GT response 提取规则：
        grundtruth = rec['response']
        response = grundtruth.split('</think>')[1]
        response = response.split('</answer>')[0]
        response = response.split('<answer>')[-1]
        response = response.replace('listener:', '')
    
    额外返回 gt_thinking，用于 _compute_metrics 的 emotion/strategy 指标。
    优先使用 rec['thinking']，没有时从 rec['response'] 的 <think> 中取。
    """
    raw = normalize_think_tags(rec.get("response", ""))

    try:
        gt_resp = raw.split("</think>")[1]
        gt_resp = gt_resp.split("</answer>")[0]
        gt_resp = gt_resp.split("<answer>")[-1]
        gt_resp = gt_resp.replace("listener:", "")
        gt_resp = gt_resp.strip()
        gt_resp = add_period_if_needed(gt_resp)
    except Exception as e:
        gt_resp = ""
        return "", str(rec.get("thinking", "") or ""), f"gt_extract_error: {type(e).__name__}: {e}"

    gt_thinking = str(rec.get("thinking", "") or "").strip()
    if not gt_thinking:
        try:
            gt_thinking = raw.split("</think>")[0]
            gt_thinking = gt_thinking.split("<think>")[-1]
            gt_thinking = gt_thinking.strip()
        except Exception:
            gt_thinking = ""

    if not gt_resp:
        return gt_resp, gt_thinking, "empty_gt_after_extraction"

    # print(gt_resp)

    return gt_resp, gt_thinking, None


def extract_pred_user_rule(rec: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
    """
    完全对齐用户 get_evaluate() 里的 Pred response 提取规则：
        grundtruth = rec['responses'][0]
        thinking = grundtruth.split('</think>')[0]
        thinking = thinking.split('<think>')[-1]
        response = grundtruth.split('</think>')[1]
        response = response.split('<answer>')[1]
        response = response.split('</answer>')[0]
        response = re.sub(r'\\(.*?\\)', '', response).strip()
        response = response.replace('listener:', '').strip()
    """
    try:
        if "responses" not in rec:
            return "", "", "missing_responses_field"

        responses = rec["responses"]
        if isinstance(responses, list):
            if len(responses) == 0:
                return "", "", "responses_empty_list"
            raw = responses[0]
        elif isinstance(responses, str):
            raw = responses
        else:
            return "", "", f"responses_invalid_type: {type(responses).__name__}"

        raw = normalize_think_tags(raw)

        pred_thinking = raw.split("</think>")[0]
        pred_thinking = pred_thinking.split("<think>")[-1]
        pred_thinking = pred_thinking.strip()

        pred_resp = raw.split("</think>")[1]
        pred_resp = pred_resp.split("<answer>")[1]
        pred_resp = pred_resp.split("</answer>")[0]
        pred_resp = re.sub(r"\(.*?\)", "", pred_resp).strip()
        pred_resp = pred_resp.replace("listener:", "").strip()
        pred_resp = add_period_if_needed(pred_resp)

        # print(pred_resp)

        if not pred_resp:
            return "", pred_thinking, "empty_pred_after_extraction"

        return pred_resp, pred_thinking, None

    except Exception as e:
        return "", "", f"pred_extract_error: {type(e).__name__}: {e}"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 解析失败: path={path}, line={line_no}, error={e}") from e
    return records


def to_jsonable(x: Any) -> Any:
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass

    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    return x


def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def append_json_list(path: str, item: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    data = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                print(f"警告: {path} 不是 list，将覆盖为 list。")
                data = []
        except Exception:
            print(f"警告: {path} 无法读取为 JSON list，将覆盖为 list。")
            data = []

    data.append(item)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=4)


def build_eval_rows(records: List[Dict[str, Any]], skip_bad: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    bad_cases: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        gt_resp, gt_thinking, gt_error = extract_gt_user_rule(rec)
        pred_resp, pred_thinking, pred_error = extract_pred_user_rule(rec)

        bad = bool(gt_error or pred_error)
        if bad:
            bad_cases.append(
                {
                    "sample_index": idx,
                    "hash_id": rec.get("hash_id", ""),
                    "conv_id": rec.get("conv_id", ""),
                    "u_id": rec.get("u_id", ""),
                    "turn_id": rec.get("turn_id", ""),
                    "source": rec.get("source", ""),
                    "emotion": rec.get("emotion", ""),
                    "gt_error": gt_error,
                    "pred_error": pred_error,
                    "raw_response_head": str(rec.get("response", ""))[:500],
                    "raw_responses_0_head": str((rec.get("responses", [""]) or [""])[0])[:500]
                    if isinstance(rec.get("responses", None), list)
                    else str(rec.get("responses", ""))[:500],
                }
            )

        if skip_bad and bad:
            continue

        rows.append(
            {
                "sample_index": idx,
                "hash_id": rec.get("hash_id", ""),
                "conv_id": rec.get("conv_id", ""),
                "u_id": rec.get("u_id", ""),
                "turn_id": rec.get("turn_id", ""),
                "source": rec.get("source", ""),
                "emotion": rec.get("emotion", ""),
                "ground_truth": gt_resp,
                "prediction": pred_resp,
                "thinking": gt_thinking,
                "generated_thinking": pred_thinking,
                "gt_error": gt_error,
                "pred_error": pred_error,
            }
        )

    return rows, bad_cases


def evaluate_subset(rows: List[Dict[str, Any]], compute_extra: bool, bart_model_path: Optional[str]) -> Dict[str, Any]:
    preds = [r["prediction"] for r in rows]
    gts = [r["ground_truth"] for r in rows]
    pred_thinkings = [r.get("generated_thinking", "") for r in rows]
    gt_thinkings = [r.get("thinking", "") for r in rows]

    metrics = _compute_metrics(
        preds,
        gts,
        pred_thinkings,
        gt_thinkings,
        compute_extra=compute_extra,
        bart_model_path=bart_model_path,
    )
    return to_jsonable(metrics)


def print_metric_line(label: str, m: Dict[str, Any]) -> None:
    extra_str = ""
    for ek in ["bartscore", "meteor", "cider"]:
        if ek in m and m[ek] is not None:
            try:
                extra_str += f"  {ek.upper()}={float(m[ek]):.4f}"
            except Exception:
                extra_str += f"  {ek.upper()}={m[ek]}"

    print(
        f"  [{label:<8}] N={int(m.get('num_samples', 0)):>5}  "
        f"BLEU-1={float(m.get('bleu1', 0) or 0):.4f}  "
        f"BLEU-4={float(m.get('bleu4', 0) or 0):.4f}  "
        f"ROUGE-L={float(m.get('rougel', 0) or 0):.4f}  "
        f"Dist-2={float(m.get('dist2', 0) or 0):.2f}  "
        f"EmoAcc={float(m.get('emotion_acc', 0) or 0):.4f}  "
        f"StrAcc={float(m.get('strategy_acc', 0) or 0):.4f}"
        + extra_str
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="生成结果 jsonl 文件路径。注意：不是 parquet。")
    parser.add_argument("--ex_name", required=True, help="实验名称。")
    parser.add_argument("--inference_path", default="inference", help="包含 run_rag_inference.py 的目录。默认 inference。")
    parser.add_argument("--bart_model", type=str, default=None, help="BART 模型路径，默认交给 _compute_metrics。")
    parser.add_argument("--no_extra", action="store_true", help="关闭 BARTScore / METEOR / CIDEr。默认开启。")
    parser.add_argument("--skip_bad", action="store_true", help="跳过 GT/Pred 提取失败的样本。默认不跳过，失败项用空字符串参与总体评估。")
    parser.add_argument("--out_json", type=str, default=None, help="保存本次聚合指标 JSON。")
    parser.add_argument("--out_bad_jsonl", type=str, default=None, help="保存提取失败样本 JSONL。")
    parser.add_argument("--append_result_json", type=str, default=None, help="把 {ex_name: metrics} 追加到指定 JSON list 文件。")
    args = parser.parse_args()

    # 延迟导入，确保 --inference_path 生效。
    sys.path.insert(0, args.inference_path)
    global _compute_metrics
    from run_rag_inference import _compute_metrics  # type: ignore

    compute_extra = not args.no_extra

    records = load_jsonl(args.input)
    if not records:
        raise ValueError(f"输入为空: {args.input}")

    rows, bad_cases = build_eval_rows(records, skip_bad=args.skip_bad)
    if not rows:
        raise ValueError("没有可评估样本。请检查提取规则或关闭 --skip_bad。")

    gt_fail = sum(1 for b in bad_cases if b.get("gt_error"))
    pred_fail = sum(1 for b in bad_cases if b.get("pred_error"))

    ed_rows = [r for r in rows if str(r.get("source", "")) == "ED"]
    es_rows = [r for r in rows if str(r.get("source", "")) == "ESConv"]

    print("\n" + "=" * 80)
    print(f"Evaluate total file: {args.ex_name}")
    print(f"Input: {args.input}")
    print(f"Raw samples: {len(records)}")
    print(f"Evaluated samples: {len(rows)}")
    print(f"GT 提取失败数量: {gt_fail}")
    print(f"Pred 提取失败数量: {pred_fail}")
    print(f"Compute BARTScore/METEOR/CIDEr: {compute_extra}")
    print("=" * 80)

    all_metrics: Dict[str, Any] = {}

    subsets = []
    if ed_rows or es_rows:
        subsets.extend([("ED", ed_rows), ("ESConv", es_rows)])
    subsets.append(("Overall", rows))

    for label, subset in subsets:
        if not subset:
            continue
        metrics = evaluate_subset(subset, compute_extra=compute_extra, bart_model_path=args.bart_model)
        all_metrics[label] = metrics
        print_metric_line(label, metrics)

    print("=" * 80)

    output_obj = {
        "name": args.ex_name,
        "input": args.input,
        "raw_samples": len(records),
        "evaluated_samples": len(rows),
        "gt_extract_failures": gt_fail,
        "pred_extract_failures": pred_fail,
        "skip_bad": args.skip_bad,
        "compute_extra": compute_extra,
        "metrics": all_metrics,
    }

    if args.out_json:
        ensure_parent_dir(args.out_json)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(output_obj), f, ensure_ascii=False, indent=2)
        print(f"Saved metrics JSON: {args.out_json}")

    if args.out_bad_jsonl:
        ensure_parent_dir(args.out_bad_jsonl)
        with open(args.out_bad_jsonl, "w", encoding="utf-8") as f:
            for b in bad_cases:
                f.write(json.dumps(to_jsonable(b), ensure_ascii=False) + "\n")
        print(f"Saved bad cases JSONL: {args.out_bad_jsonl}")

    if args.append_result_json:
        append_json_list(args.append_result_json, {args.ex_name: all_metrics})
        print(f"Appended result JSON list: {args.append_result_json}")


if __name__ == "__main__":
    main()
