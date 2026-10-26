#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json
import os
import random
import re
from typing import Any, Dict, List, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import datasets
import numpy as np
from tqdm import tqdm

# ============================================================
# 调用你的检索模块
# ============================================================

from build_ed_faiss import (
    Qwen3Embedding,
    load_faiss_resources,
    calc_total_score,
    MODEL_NAME,
    DEVICE,
)

# ============================================================
# JSONL
# ============================================================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON 解析失败: path={path}, line={line_no}, error={e}"
                ) from e

    return records


# ============================================================
# 标签 / response 处理
# ============================================================

def strip_outer_tag(text: str, tag: str) -> str:
    if not text:
        return ""

    m = re.match(
        rf"^\s*<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>\s*$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if m:
        return m.group(1).strip()

    return text.strip()


def extract_tag_value(text: Any, tag: str) -> str:
    """
    兼容：
        <Context>xxx</Context>
    以及：
        <Context> xxx
        <Emotion> xxx
        <Strategy> xxx
    """

    if text is None:
        return ""

    text = str(text).strip()

    if not text:
        return ""

    # 有闭合标签
    m = re.search(
        rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if m:
        return m.group(1).strip()

    # 无闭合标签
    m = re.search(
        rf"<{re.escape(tag)}>\s*:?\s*(.*?)"
        rf"(?=\s*<(?:Context|Summary|summary|Emotion|Strategy|Action|Response|answer|thinking|think|rag_needed)>|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if m:
        return m.group(1).strip()

    return ""


def get_raw_thinking(rec: Dict[str, Any]) -> str:
    """
    优先 thinking 字段；
    否则从 response / answer / output 中解析。
    """

    thinking = str(rec.get("thinking", "") or "").strip()

    if thinking:
        thinking = strip_outer_tag(thinking, "thinking")
        thinking = strip_outer_tag(thinking, "think")
        return thinking

    for key in ["response", "answer", "output"]:
        value = str(rec.get(key, "") or "").strip()

        if not value:
            continue

        thinking_value = extract_tag_value(value, "thinking")

        if thinking_value:
            return thinking_value

        thinking_value = extract_tag_value(value, "think")

        if thinking_value:
            return thinking_value

    # 兼容 reward_model["ground_truth"]
    reward_model = rec.get("reward_model")

    if isinstance(reward_model, dict):
        gt = str(
            reward_model.get("ground_truth", "") or ""
        ).strip()

        if gt:
            thinking_value = extract_tag_value(
                gt,
                "thinking",
            )

            if thinking_value:
                return thinking_value

            return gt

    return ""


def parse_query_information(
    rec: Dict[str, Any],
) -> Tuple[str, str, str]:
    """
    检索时需要：
        Context
        Emotion
        Strategy

    Strategy 只用于检索，不会写进最终 response。
    """

    raw = get_raw_thinking(rec)

    context = extract_tag_value(
        raw,
        "Context",
    )

    emotion = extract_tag_value(
        raw,
        "Emotion",
    )

    strategy = extract_tag_value(
        raw,
        "Strategy",
    )

    # 兼容字段直接在外层的情况
    if not emotion:
        emotion = str(
            rec.get("emotion", "") or ""
        ).strip()

    if not strategy:
        strategy = str(
            rec.get("strategy", "") or ""
        ).strip()

    return (
        context.strip(),
        emotion.strip(),
        strategy.strip(),
    )


def clean_thinking_without_strategy(
    rec: Dict[str, Any],
) -> str:
    """
    最终测试 response 删除 Strategy。

    例如：
        <Context> ...
        <Emotion> sadness
        <Strategy> self-disclosure

    变为：
        <Context> ...
        <Emotion> sadness
    """

    raw = get_raw_thinking(rec)

    if not raw:
        return ""

    raw = re.sub(
        r"\s*<Strategy>\s*:?.*$",
        "",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return raw.strip()


def extract_answer(rec: Dict[str, Any]) -> str:
    """
    提取当前测试样本真实 answer。
    """

    for key in ["answer", "output", "response"]:
        value = str(
            rec.get(key, "") or ""
        ).strip()

        if not value:
            continue

        answer = extract_tag_value(
            value,
            "answer",
        )

        if answer:
            return answer

        lower = value.lower()

        # 防止把 thinking + answer 整段当成 answer
        if (
            "<thinking>" in lower
            or "<think>" in lower
        ):
            continue

        return value

    return ""


def build_clean_response(
    rec: Dict[str, Any],
) -> str:
    """
    输出：
        <thinking>...</thinking><answer>...</answer>

    Strategy 不进入 response。
    """

    thinking = clean_thinking_without_strategy(
        rec
    )

    answer = extract_answer(
        rec
    )

    return (
        f"<thinking>{thinking}</thinking>"
        f"<answer>{answer}</answer>"
    )


# ============================================================
# conversations / 当前 dialogue
# ============================================================

def get_turn_role(
    turn: Dict[str, Any],
) -> str:

    role = (
        turn.get("role", "")
        or turn.get("from", "")
        or turn.get("speaker", "")
        or ""
    )

    role = str(
        role
    ).strip().lower()

    if role in {
        "human",
        "user",
        "speaker",
    }:
        return "user"

    if role in {
        "gpt",
        "assistant",
        "bot",
        "listener",
    }:
        return "assistant"

    return role


def get_turn_content(
    turn: Dict[str, Any],
) -> str:

    if turn.get("content") is not None:
        value = turn.get("content")

    elif turn.get("value") is not None:
        value = turn.get("value")

    else:
        value = turn.get(
            "text",
            "",
        )

    return str(
        value or ""
    ).strip()


def normalize_chat_turn(
    turn: Dict[str, Any],
) -> Dict[str, str]:

    return {
        "content": get_turn_content(turn),
        "role": get_turn_role(turn),
    }


def get_conversations(
    rec: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    优先 conversations；
    其次 prompt=list；
    最后兼容字符串 prompt/input。
    """

    conversations = rec.get(
        "conversations"
    )

    if (
        isinstance(conversations, list)
        and conversations
    ):
        result = []

        for turn in conversations:
            if not isinstance(turn, dict):
                continue

            norm = normalize_chat_turn(
                turn
            )

            if norm["content"]:
                result.append(norm)

        return result

    prompt = rec.get("prompt")

    if isinstance(prompt, list) and prompt:
        result = []

        for turn in prompt:
            if not isinstance(turn, dict):
                continue

            norm = normalize_chat_turn(
                turn
            )

            if norm["content"]:
                result.append(norm)

        return result

    text = ""

    if isinstance(
        rec.get("input"),
        str,
    ):
        text = rec.get(
            "input",
            "",
        ).strip()

    elif isinstance(
        rec.get("prompt"),
        str,
    ):
        text = rec.get(
            "prompt",
            "",
        ).strip()

    if text:
        return [
            {
                "content": text,
                "role": "user",
            }
        ]

    return []


def add_role_prefix(
    turn: Dict[str, Any],
) -> str:

    role = get_turn_role(
        turn
    )

    content = get_turn_content(
        turn
    )

    if not content:
        return ""

    lower = content.lower().strip()

    if (
        lower.startswith("speaker:")
        or lower.startswith("listener:")
    ):
        return content

    if role == "user":
        return f"speaker: {content}"

    if role == "assistant":
        return f"listener: {content}"

    return content


def format_conversation(
    conversations: List[Dict[str, Any]],
) -> str:

    lines = []

    for turn in conversations:

        if not isinstance(turn, dict):
            continue

        line = add_role_prefix(
            turn
        )

        if line:
            lines.append(line)

    return "\n".join(
        lines
    ).strip()


def get_current_dialogue(
    rec: Dict[str, Any],
) -> str:

    conversations = get_conversations(
        rec
    )

    text = format_conversation(
        conversations
    )

    if text:
        return text

    for key in [
        "input",
        "prompt",
    ]:
        value = rec.get(key)

        if (
            isinstance(value, str)
            and value.strip()
        ):
            return value.strip()

    return ""


# ============================================================
# 当前样本 / 候选过滤
# ============================================================

def to_str_id(x: Any) -> str:
    if x is None:
        return ""

    return str(
        x
    ).strip()


def same_sample(
    current_rec: Dict[str, Any],
    candidate_meta: Dict[str, Any],
) -> bool:
    """
    你的 ED-database-meta.pkl 当前至少保存 hash_id，
    因此优先过滤相同 hash_id。
    """

    cur_hash = to_str_id(
        current_rec.get("hash_id")
    )

    cand_hash = to_str_id(
        candidate_meta.get("hash_id")
    )

    if (
        cur_hash
        and cand_hash
        and cur_hash == cand_hash
    ):
        return True

    return False


# ============================================================
# Semantic + Emotion + Strategy 检索
# ============================================================

def retrieve_candidates_rerank(
    rec: Dict[str, Any],
    query_context: str,
    query_action: str,
    query_emotion: str,
    query_strategy: str,
    embedder: Qwen3Embedding,
    index,
    metadata: List[Dict[str, Any]],
    coarse_top: int = 50,
    top_k: int = 1,
) -> List[Dict[str, Any]]:
    """
    对应 ED_retrieval.py 中原来的：

        query_text
            -> FAISS coarse retrieval
            -> Semantic score
            -> Emotion score
            -> Strategy score
            -> total score rerank

    这里只是把 index / metadata 放在外面一次性加载，
    避免每条测试样本都重新从磁盘读 index。
    """

    query_text = (
        f"{query_context}\n"
        f"{query_action}"
    ).strip()

    if not query_text:
        return []

    q_emb = embedder.encode(
        [query_text],
        normalize_embeddings=True,
    ).astype(np.float32)

    # 为了过滤自己，多取一些
    search_n = min(
        max(
            coarse_top,
            top_k + 20,
        ),
        index.ntotal,
    )

    distances, indices = index.search(
        q_emb,
        search_n,
    )

    candidates = []

    for l2_dist, meta_idx in zip(
        distances[0],
        indices[0],
    ):

        if meta_idx < 0:
            continue

        meta_idx = int(
            meta_idx
        )

        if meta_idx >= len(metadata):
            continue

        meta = metadata[
            meta_idx
        ]

        if not isinstance(meta, dict):
            continue

        if same_sample(
            rec,
            meta,
        ):
            continue

        score_info = calc_total_score(
            faiss_distance=float(l2_dist),
            q_emotion=query_emotion,
            c_emotion=str(
                meta.get(
                    "emotion",
                    "",
                )
                or ""
            ),
            q_strategy=query_strategy,
            c_strategy=str(
                meta.get(
                    "strategy",
                    "",
                )
                or ""
            ),
        )

        candidate = {
            "meta": meta,
            "memory_index": meta_idx,
            **score_info,
        }

        candidates.append(
            candidate
        )

    # 真正按照 total_score rerank
    candidates.sort(
        key=lambda x: x["total_score"],
        reverse=True,
    )

    return candidates[
        :top_k
    ]


# ============================================================
# RelevantExperience
# ============================================================

def parse_context_from_example(
    example: str,
) -> str:
    """
    你当前 ED_retrieval.py 的 metadata 中没有直接保存 context，
    但 example 格式是：

        Context: xxx Emotion: xxx. Action: xxx Answer: xxx

    因此这里兼容地从 example 中抽取 Context。

    如果以后 metadata.pkl 里直接保存 context，
    会优先使用 meta["context"]。
    """

    example = str(
        example or ""
    ).strip()

    if not example:
        return ""

    m = re.search(
        r"Context:\s*(.*?)\s+Emotion:\s*",
        example,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if m:
        return m.group(1).strip()

    return ""


def get_candidate_context(
    meta: Dict[str, Any],
) -> str:
    """
    优先读取 metadata 中真正的 context；
    没有时从 example 里解析。
    """

    context = str(
        meta.get(
            "context",
            "",
        )
        or ""
    ).strip()

    if context:
        return context

    return parse_context_from_example(
        meta.get(
            "example",
            "",
        )
    )


def get_candidate_summary(
    meta: Dict[str, Any],
) -> str:
    """
    如果 preprocessing 后的 metadata 中保存了 abstract，
    就直接使用。

    当前原始 ED_retrieval.py 没有把 abstract 放进 metadata，
    所以退化为候选 Context。
    """

    summary = str(
        meta.get(
            "abstract",
            "",
        )
        or ""
    ).strip()

    if summary:
        return summary

    return get_candidate_context(
        meta
    )


def format_single_relevant_experience(
    candidate: Dict[str, Any],
) -> str:
    """
    与你刚上传的 ED test 脚本保持一致：
    RelevantExperience 中不放 Strategy。
    """

    meta = candidate[
        "meta"
    ]

    context = (
        get_candidate_context(meta)
        or "None"
    )

    summary = (
        get_candidate_summary(meta)
        or "None"
    )

    emotion = str(
        meta.get(
            "emotion",
            "",
        )
        or "None"
    ).strip()

    answer = str(
        meta.get(
            "answer",
            "",
        )
        or "None"
    ).strip()

    return (
        "<RelevantExperience>\n"
        "<Context>\n"
        + context
        + "\n"
        "<summary>\n"
        + summary
        + "\n"
        "<Emotion>\n"
        + emotion
        + "\n"
        "<answer>\n"
        + answer
        + "\n"
        "</RelevantExperience>"
    )


def format_retrieved(
    candidates: List[Dict[str, Any]],
) -> str:

    if not candidates:
        return (
            "<RelevantExperience>\n"
            "None\n"
            "</RelevantExperience>"
        )

    return "\n\n".join(
        format_single_relevant_experience(
            candidate
        )
        for candidate in candidates
    ).strip()


def build_rag_prompt_text(
    rec: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> str:

    relevant = format_retrieved(
        candidates
    )

    current_dialogue = get_current_dialogue(
        rec
    )

    return (
        relevant
        + "\n\n"
        + "<CurrentDialogue>\n"
        + current_dialogue
        + "\n</CurrentDialogue>"
    )


def build_no_rag_prompt_text(
    rec: Dict[str, Any],
) -> str:

    current_dialogue = get_current_dialogue(
        rec
    )

    return (
        "<RelevantExperience>\n"
        "None\n"
        "</RelevantExperience>\n\n"
        "<CurrentDialogue>\n"
        + current_dialogue
        + "\n</CurrentDialogue>"
    )


def build_no_rag_prompt(
    rec: Dict[str, Any],
) -> List[Dict[str, str]]:

    return [
        {
            "content": build_no_rag_prompt_text(
                rec
            ),
            "role": "user",
        }
    ]


def build_rag_prompt(
    rec: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, str]]:

    return [
        {
            "content": build_rag_prompt_text(
                rec,
                candidates,
            ),
            "role": "user",
        }
    ]


# ============================================================
# 参数
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default=(
            "./ED_test.jsonl"
        ),
        help="ED test JSONL",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出 parquet 完整路径",
    )

    parser.add_argument(
        "--rag_ratio",
        type=float,
        default=0.15,
        help="随机使用 RAG 的样本比例，默认 0.15",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=43,
    )

    parser.add_argument(
        "--coarse_top",
        type=int,
        default=50,
        help="FAISS 先进行多少个候选的粗召回",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="最终加入几个 RelevantExperience",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_NAME,
        help="默认直接读取 ED_retrieval.py 中 MODEL_NAME",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        choices=[
            "cuda",
            "cpu",
        ],
    )

    parser.add_argument(
        "--drop_aux_fields",
        action="store_true",
        help="只输出 prompt / response",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if not (
        0.0
        <= args.rag_ratio
        <= 1.0
    ):
        raise ValueError(
            "--rag_ratio 必须在 [0,1]"
        )

    if args.top_k <= 0:
        raise ValueError(
            "--top_k 必须 > 0"
        )

    # ========================================================
    # 读取 Test
    # ========================================================

    print(
        "\n=== 加载 ED Test ==="
    )

    print(
        "data_path:",
        args.data_path,
    )

    records = read_jsonl(
        args.data_path
    )

    if not records:
        raise ValueError(
            "输入测试集为空。"
        )

    n = len(
        records
    )

    rag_count = int(
        round(
            n
            * args.rag_ratio
        )
    )

    rng = random.Random(
        args.seed
    )

    rag_indices = (
        set(
            rng.sample(
                range(n),
                rag_count,
            )
        )
        if rag_count > 0
        else set()
    )

    print(
        "测试样本数:",
        n,
    )

    print(
        "RAG 比例:",
        args.rag_ratio,
    )

    print(
        "随机种子:",
        args.seed,
    )

    print(
        "随机选中 RAG:",
        len(rag_indices),
    )

    # ========================================================
    # 一次性加载检索资源
    # ========================================================

    index = None
    metadata = None
    embedder = None

    if rag_indices:

        print(
            "\n=== 加载 FAISS / Metadata ==="
        )

        index, metadata = (
            load_faiss_resources()
        )

        print(
            "index.ntotal:",
            index.ntotal,
        )

        print(
            "metadata:",
            len(metadata),
        )

        if (
            index.ntotal
            != len(metadata)
        ):
            print(
                "WARNING: "
                "index.ntotal != len(metadata)"
            )

        print(
            "\n=== 加载 Embedding Model ==="
        )

        print(
            "model_path:",
            args.model_path,
        )

        embedder = Qwen3Embedding(
            args.model_path,
            args.device,
        )

    # ========================================================
    # 开始处理
    # ========================================================

    rows = []

    successful_rag = 0
    no_query_count = 0
    no_candidate_count = 0
    empty_answer_count = 0
    missing_emotion_count = 0
    missing_strategy_count = 0

    for idx, rec in enumerate(
        tqdm(
            records,
            desc="ED test rerank RAG",
        )
    ):

        row = dict(
            rec
        )

        # ----------------------------------------------------
        # 最终 response 删除 Strategy
        # ----------------------------------------------------

        row["response"] = (
            build_clean_response(
                rec
            )
        )

        if not extract_answer(
            rec
        ):
            empty_answer_count += 1

        # ----------------------------------------------------
        # 默认先生成 None RAG
        # ----------------------------------------------------

        row["prompt"] = (
            build_no_rag_prompt(
                rec
            )
        )

        row["sample_type"] = (
            "test_no_rag"
        )

        # 保持 schema 稳定
        row["retrieval_score"] = None
        row["semantic_score"] = None
        row["emotion_score"] = None
        row["strategy_score"] = None
        row["retrieved_memory_index"] = None
        row["retrieved_hash_id"] = None

        # ----------------------------------------------------
        # 没有随机抽中
        # ----------------------------------------------------

        if idx not in rag_indices:

            if args.drop_aux_fields:
                row = {
                    "prompt": row["prompt"],
                    "response": row["response"],
                }

            rows.append(
                row
            )

            continue

        # ----------------------------------------------------
        # 从当前测试样本中提取：
        # Context / Emotion / Strategy
        # ----------------------------------------------------

        (
            abstract_context,
            query_emotion,
            query_strategy,
        ) = parse_query_information(
            rec
        )

        current_dialogue = (
            get_current_dialogue(
                rec
            )
        )

        # 和数据库 preprocessing 保持对应：
        #
        # database context =
        #   abstract context + prompt
        #
        # query context =
        #   abstract context + current dialogue
        # ----------------------------------------------------

        if abstract_context:

            query_context = (
                abstract_context
                + " "
                + current_dialogue
            ).strip()

        else:

            query_context = (
                current_dialogue
            )

        # 你的数据库 action 就是 Strategy，
        # 所以 query_action 直接使用 Strategy
        query_action = (
            query_strategy
        )

        if not query_context:
            no_query_count += 1

            row["sample_type"] = (
                "test_rag_selected_but_no_query"
            )

        elif not query_emotion:
            missing_emotion_count += 1

            row["sample_type"] = (
                "test_rag_selected_but_no_emotion"
            )

        elif not query_strategy:
            missing_strategy_count += 1

            row["sample_type"] = (
                "test_rag_selected_but_no_strategy"
            )

        else:

            # ------------------------------------------------
            # Semantic + Emotion + Strategy Rerank
            # ------------------------------------------------

            candidates = (
                retrieve_candidates_rerank(
                    rec=rec,
                    query_context=query_context,
                    query_action=query_action,
                    query_emotion=query_emotion,
                    query_strategy=query_strategy,
                    embedder=embedder,
                    index=index,
                    metadata=metadata,
                    coarse_top=args.coarse_top,
                    top_k=args.top_k,
                )
            )

            # ------------------------------------------------
            # 成功
            # ------------------------------------------------

            if candidates:

                row["prompt"] = (
                    build_rag_prompt(
                        rec,
                        candidates,
                    )
                )

                row["sample_type"] = (
                    "test_random_rag"
                )

                first = candidates[
                    0
                ]

                row["retrieval_score"] = (
                    float(
                        first[
                            "total_score"
                        ]
                    )
                )

                row["semantic_score"] = (
                    float(
                        first[
                            "semantic_score"
                        ]
                    )
                )

                row["emotion_score"] = (
                    float(
                        first[
                            "emotion_score"
                        ]
                    )
                )

                row["strategy_score"] = (
                    float(
                        first[
                            "strategy_score"
                        ]
                    )
                )

                row["retrieved_memory_index"] = (
                    int(
                        first[
                            "memory_index"
                        ]
                    )
                )

                row["retrieved_hash_id"] = (
                    first[
                        "meta"
                    ].get(
                        "hash_id"
                    )
                )

                successful_rag += 1

            else:

                no_candidate_count += 1

                row["sample_type"] = (
                    "test_rag_selected_but_no_candidate"
                )

        # ----------------------------------------------------
        # 可选只保留 prompt + response
        # ----------------------------------------------------

        if args.drop_aux_fields:

            row = {
                "prompt": row["prompt"],
                "response": row["response"],
            }

        rows.append(
            row
        )

    # ========================================================
    # 保存 parquet
    # ========================================================

    output_dir = os.path.dirname(
        os.path.abspath(
            args.output_path
        )
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    ds = datasets.Dataset.from_list(
        rows
    )

    ds.to_parquet(
        args.output_path
    )

    # ========================================================
    # 统计
    # ========================================================

    print(
        "\n=== 保存完成 ==="
    )

    print(
        "输出路径:",
        args.output_path,
    )

    print(
        "输入样本数:",
        n,
    )

    print(
        "输出样本数:",
        len(rows),
    )

    print(
        "随机选中 RAG:",
        len(rag_indices),
    )

    print(
        "成功加入 RAG:",
        successful_rag,
    )

    print(
        "无 query:",
        no_query_count,
    )

    print(
        "缺 Emotion:",
        missing_emotion_count,
    )

    print(
        "缺 Strategy:",
        missing_strategy_count,
    )

    print(
        "无合法候选:",
        no_candidate_count,
    )

    print(
        "answer 为空:",
        empty_answer_count,
    )

    print(
        "实际成功 RAG 比例:",
        f"{successful_rag / n:.4%}",
    )

    print(
        "\n=== parquet 字段 ==="
    )

    print(
        ds.column_names
    )

    # ========================================================
    # 抽查普通样本
    # ========================================================

    if not args.drop_aux_fields:

        print(
            "\n=== 第一条普通样本 ==="
        )

        for r in rows:

            if (
                r.get(
                    "sample_type"
                )
                == "test_no_rag"
            ):

                print(
                    json.dumps(
                        r["prompt"],
                        ensure_ascii=False,
                        indent=2,
                    )[:3000]
                )

                print(
                    "\nresponse:"
                )

                print(
                    r["response"][
                        :1000
                    ]
                )

                break

        # ====================================================
        # 抽查 RAG
        # ====================================================

        print(
            "\n=== 第一条 RAG 样本 ==="
        )

        found = False

        for r in rows:

            if (
                r.get(
                    "sample_type"
                )
                == "test_random_rag"
            ):

                print(
                    "retrieval_score:",
                    r.get(
                        "retrieval_score"
                    ),
                )

                print(
                    "semantic_score:",
                    r.get(
                        "semantic_score"
                    ),
                )

                print(
                    "emotion_score:",
                    r.get(
                        "emotion_score"
                    ),
                )

                print(
                    "strategy_score:",
                    r.get(
                        "strategy_score"
                    ),
                )

                print(
                    "retrieved_hash_id:",
                    r.get(
                        "retrieved_hash_id"
                    ),
                )

                print(
                    "\nprompt:"
                )

                print(
                    json.dumps(
                        r["prompt"],
                        ensure_ascii=False,
                        indent=2,
                    )[:5000]
                )

                print(
                    "\nresponse:"
                )

                print(
                    r["response"][
                        :1000
                    ]
                )

                found = True

                break

        if not found:

            print(
                "没有成功构造 RAG 样本。"
            )


if __name__ == "__main__":
    main()
