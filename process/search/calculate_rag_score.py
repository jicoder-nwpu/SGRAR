#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import argparse

from pathlib import Path
from collections import Counter


# ============================================================
# 1. 情绪分数
# ============================================================

EMOTION_VAD_TARGET = {
    "surprised": {"vad": [0.568, 0.71, 0.078], "mean": 0.5107},
    "excited": {"vad": [0.816, 0.862, 0.418], "mean": 0.4380},
    "angry": {"vad": [-0.756, 0.66, 0.208], "mean": 0.7013},
    "proud": {"vad": [0.812, 0.4, 0.746], "mean": 0.3070},
    "sad": {"vad": [-0.55, -0.334, -0.702], "mean": 0.6530},
    "annoyed": {"vad": [-0.792, 0.566, -0.31], "mean": 0.7780},
    "grateful": {"vad": [0.916, -0.294, 0.12], "mean": 0.2783},
    "lonely": {"vad": [-0.5, -0.548, -0.524], "mean": 0.5793},
    "afraid": {"vad": [-0.977, 0.352, -0.529], "mean": 0.8097},
    "terrified": {"vad": [-0.82, 0.804, -0.226], "mean": 0.8083},
    "guilty": {"vad": [-0.73, 0.54, -0.296], "mean": 0.7610},
    "impressed": {"vad": [0.612, 0.54, -0.142], "mean": 0.5117},
    "disgusted": {"vad": [-0.898, 0.546, -0.452], "mean": 0.8160},
    "hopeful": {"vad": [0.894, -0.286, 0.254], "mean": 0.2610},
    "confident": {"vad": [0.53, -0.352, 0.446], "mean": 0.2787},
    "furious": {"vad": [-0.876, 0.906, 0.196], "mean": 0.7643},
    "anxious": {"vad": [-0.438, 0.75, -0.132], "mean": 0.7200},
    "anticipating": {"vad": [0.244, -0.062, 0.334], "mean": 0.3933},
    "joyful": {"vad": [0.99, 0.24, 0.296], "mean": 0.3257},
    "nostalgic": {"vad": [-0.084, -0.298, -0.632], "mean": 0.5697},
    "disappointed": {"vad": [-0.858, -0.056, -0.518], "mean": 0.7200},
    "prepared": {"vad": [0.42, -0.04, 0.466], "mean": 0.3457},
    "jealous": {"vad": [-0.654, 0.71, -0.31], "mean": 0.7790},
    "content": {"vad": [0.528, -0.408, 0.118], "mean": 0.3243},
    "devastated": {"vad": [-0.76, 0.088, -0.472], "mean": 0.7200},
    "embarrassed": {"vad": [-0.632, 0.12, -0.47], "mean": 0.7037},
    "caring": {"vad": [0.27, -0.062, 0.0], "mean": 0.4447},
    "sentimental": {"vad": [0.166, -0.244, -0.376], "mean": 0.4943},
    "trusting": {"vad": [0.714, 0.016, 0.5], "mean": 0.3003},
    "ashamed": {"vad": [-0.688, 0.176, -0.544], "mean": 0.7347},
    "apprehensive": {"vad": [-0.188, 0.184, -0.138], "mean": 0.5850},
    "faithful": {"vad": [0.776, -0.334, 0.254], "mean": 0.2727},

    "guilt": {"vad": [-0.584, 0.134, -0.588], "mean": 0.6843},
    "anger": {"vad": [-0.666, 0.73, 0.314], "mean": 0.6600},
    "nervousness": {"vad": [-0.674, 0.83, -0.518], "mean": 0.7903},
    "fear": {"vad": [-0.854, 0.68, -0.414], "mean": 0.8080},
    "pain": {"vad": [-0.918, 0.53, -0.351], "mean": 0.8015},
    "anxiety": {"vad": [-0.708, 0.73, -0.316], "mean": 0.7623},
    "disgust": {"vad": [-0.896, 0.55, -0.366], "mean": 0.8020},
    "jealousy": {"vad": [-0.708, 0.566, -0.038], "mean": 0.7187},
    "depression": {"vad": [-0.938, 0.04, -0.536], "mean": 0.7423},
    "sadness": {"vad": [-0.896, -0.424, -0.672], "mean": 0.6840},
    "shame": {"vad": [-0.88, 0.34, -0.69], "mean": 0.7850},
}


# ============================================================
# 2. 标准策略及分数
# ============================================================

STRATEGY_LEVEL_MAP = {

    # --------------------------------------------------------
    # Level 1
    # --------------------------------------------------------

    "gratitude prompting": {
        "level": 1,
        "score": 0.0
    },

    "restatement or paraphrasing": {
        "level": 1,
        "score": 0.0
    },

    "others": {
        "level": 1,
        "score": 0.0
    },

    "information": {
        "level": 1,
        "score": 0.0
    },

    "question": {
        "level": 1,
        "score": 0.0
    },

    # --------------------------------------------------------
    # Level 2
    # --------------------------------------------------------

    "neutral validation": {
        "level": 2,
        "score": 0.5
    },

    "positive reinforcement": {
        "level": 2,
        "score": 0.5
    },

    "self-disclosure": {
        "level": 2,
        "score": 0.5
    },

    "exploring actions and intentions": {
        "level": 2,
        "score": 0.5
    },

    "affirmation and reassurance": {
        "level": 2,
        "score": 0.5
    },

    # --------------------------------------------------------
    # Level 3
    # --------------------------------------------------------

    "reflection of feelings": {
        "level": 3,
        "score": 1.0
    },

    "cognitive reframing": {
        "level": 3,
        "score": 1.0
    },

    "exploring feelings and emotions": {
        "level": 3,
        "score": 1.0
    },

    "exploring thoughts and cognition": {
        "level": 3,
        "score": 1.0
    },

    "providing suggestions": {
        "level": 3,
        "score": 1.0
    },
}


# ============================================================
# 3. ED 策略名称 -> 标准策略名称
#
# 根据你的 3008 条 warning 总结出来
# ============================================================

STRATEGY_ALIAS_MAP = {

    # --------------------------------------------------------
    # Emotion-focused
    # --------------------------------------------------------

    "labeling":
        "reflection of feelings",

    "mirroring":
        "reflection of feelings",

    "soothing":
        "affirmation and reassurance",

    # --------------------------------------------------------
    # Content-focused
    # --------------------------------------------------------

    "sharing experience":
        "self-disclosure",

    "restating":
        "restatement or paraphrasing",

    "reframing":
        "cognitive reframing",

    # --------------------------------------------------------
    # Action-focused
    # --------------------------------------------------------

    "collaborative exploration":
        "exploring actions and intentions",

    "concrete suggestion":
        "providing suggestions",

    "encouraging expression":
        "exploring feelings and emotions",

    "resource directing":
        "providing suggestions",
}



GENERIC_STRATEGY_CATEGORIES = {
    "emotion-focused",
    "content-focused",
    "action-focused",
}


# ============================================================
# 5. sample_type 分数
# ============================================================

SAMPLE_TYPE_SCORE = {
    "test_random_rag": 1.0,
    "test_no_rag": 0.0,
}



UNMATCHED_STRATEGY_DEFAULT_SCORE = 0.0


# ============================================================
# 7. 标准化文本
# ============================================================

def normalize_text(text):

    if text is None:
        return None

    text = str(text)

    text = text.strip()

    text = text.lower()

    # Markdown 加粗
    text = text.replace("**", "")

    # 一些特殊字符
    text = text.replace("_comma_", ",")

    # fancy dash
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    # 多空格
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 8. 提取 <Strategy> 内容
# ============================================================

def extract_strategy(thinking):
    """
    不再强行只提取一个短词。

    而是先提取整个 <Strategy> 内容。

    例如：

    <Strategy>
    I would use labeling to acknowledge...
    
    返回：

    i would use labeling to acknowledge...

    后面再从这个文本中识别真正策略。
    """

    if not thinking:
        return None

    thinking = str(thinking)

    # --------------------------------------------------------
    # 找 <Strategy>
    # --------------------------------------------------------

    match = re.search(
        r"<Strategy>\s*(.*?)"
        r"(?="
        r"</Strategy>"
        r"|<Context>"
        r"|<Emotion>"
        r"|<answer>"
        r"|<Answer>"
        r"|$"
        r")",
        thinking,
        flags=(
            re.IGNORECASE
            | re.DOTALL
        )
    )

    if not match:
        return None

    strategy_text = (
        match.group(1)
    )

    strategy_text = normalize_text(
        strategy_text
    )

    if not strategy_text:
        return None

    return strategy_text


# ============================================================
# 9. 判断是否只有大类
# ============================================================

def detect_generic_category(text):

    if not text:
        return None

    text = normalize_text(text)

    # --------------------------------------------------------
    # 完全等于大类
    # --------------------------------------------------------

    cleaned = text.strip(
        " [](){}:;,.!-"
    )

    if cleaned in GENERIC_STRATEGY_CATEGORIES:
        return cleaned

    # --------------------------------------------------------
    # 例如：
    #
    # [emotion-focused
    # emotion-focused]
    # (emotion-focused
    #
    # --------------------------------------------------------

    for category in GENERIC_STRATEGY_CATEGORIES:

        pattern = (
            r"^[\[\(\s]*"
            + re.escape(category)
            + r"[\]\)\s.,;:-]*$"
        )

        if re.match(
            pattern,
            text
        ):
            return category

    return None


# ============================================================
# 10. 匹配 Strategy
# ============================================================

def match_strategy(extracted_strategy):
    """
    返回：

    matched_strategy
    detected_alias
    match_type

    match_type:
        standard_exact
        alias_exact
        alias_in_text
        standard_in_text
        category_only
        unmatched
        missing
    """

    if not extracted_strategy:

        return (
            None,
            None,
            "missing"
        )

    text = normalize_text(
        extracted_strategy
    )

    # ========================================================
    # 1. 标准策略完全匹配
    # ========================================================

    cleaned = text.strip(
        " [](){}:;,.!-"
    )

    if cleaned in STRATEGY_LEVEL_MAP:

        return (
            cleaned,
            None,
            "standard_exact"
        )

    # ========================================================
    # 2. Alias 完全匹配
    # ========================================================

    if cleaned in STRATEGY_ALIAS_MAP:

        standard = (
            STRATEGY_ALIAS_MAP[
                cleaned
            ]
        )

        return (
            standard,
            cleaned,
            "alias_exact"
        )



    candidates = []

    # --------------------------------------------------------
    # Alias
    # --------------------------------------------------------

    for alias, standard in (
        STRATEGY_ALIAS_MAP.items()
    ):

        index = text.find(
            alias
        )

        if index != -1:

            candidates.append(
                (
                    index,
                    -len(alias),
                    "alias",
                    alias,
                    standard
                )
            )

    # --------------------------------------------------------
    # 标准策略
    # --------------------------------------------------------

    for standard in STRATEGY_LEVEL_MAP:

        index = text.find(
            standard
        )

        if index != -1:

            candidates.append(
                (
                    index,
                    -len(standard),
                    "standard",
                    None,
                    standard
                )
            )

    # --------------------------------------------------------
    # 有具体策略
    # --------------------------------------------------------

    if candidates:

        # 最早出现的策略优先
        # 同位置则更长的关键词优先

        candidates.sort()

        (
            _,
            _,
            source,
            alias,
            standard
        ) = candidates[0]

        if source == "alias":

            return (
                standard,
                alias,
                "alias_in_text"
            )

        else:

            return (
                standard,
                None,
                "standard_in_text"
            )

    # ========================================================
    # 4. 没有找到具体策略，
    #    看是不是只有大类
    # ========================================================

    category = detect_generic_category(
        text
    )

    if category:

        return (
            None,
            category,
            "category_only"
        )

    # ========================================================
    # 5. 彻底无法匹配
    # ========================================================

    return (
        None,
        None,
        "unmatched"
    )


# ============================================================
# 11. Warning 保存
# ============================================================

def add_warning(
    warning_list,
    warning_type,
    item,
    line_number,
    message,
    extracted_strategy=None,
    detected_alias=None,
    match_type=None,
):

    warning = {

        "warning_type":
            warning_type,

        "line_number":
            line_number,

        "hash_id":
            item.get("hash_id"),

        "conv_id":
            item.get("conv_id"),

        "u_id":
            item.get("u_id"),

        "emotion":
            item.get("emotion"),

        "sample_type":
            item.get("sample_type"),

        "extracted_strategy":
            extracted_strategy,

        "detected_alias":
            detected_alias,

        "match_type":
            match_type,

        "thinking":
            item.get(
                "thinking",
                ""
            ),

        "message":
            message,
    }

    warning_list.append(
        warning
    )


# ============================================================
# 12. 处理单条数据
# ============================================================

def process_item(
    item,
    threshold,
    warning_list,
    line_number
):

    # ========================================================
    # sample_type
    # ========================================================

    sample_type = normalize_text(
        item.get(
            "sample_type",
            ""
        )
    )

    if sample_type in SAMPLE_TYPE_SCORE:

        sample_score = (
            SAMPLE_TYPE_SCORE[
                sample_type
            ]
        )

    else:

        sample_score = 0.0

        message = (
            f"未知 sample_type: "
            f"{sample_type}"
        )

        add_warning(
            warning_list,
            "unknown_sample_type",
            item,
            line_number,
            message
        )

    # ========================================================
    # emotion
    # ========================================================

    emotion = normalize_text(
        item.get(
            "emotion",
            ""
        )
    )

    if emotion in EMOTION_VAD_TARGET:

        emotion_score = (
            EMOTION_VAD_TARGET[
                emotion
            ]["mean"]
        )

    else:

        emotion_score = 0.0

        message = (
            f"未知 emotion: "
            f"{emotion}"
        )

        add_warning(
            warning_list,
            "unknown_emotion",
            item,
            line_number,
            message
        )

    # ========================================================
    # Strategy
    # ========================================================

    thinking = item.get(
        "thinking",
        ""
    )

    extracted_strategy = (
        extract_strategy(
            thinking
        )
    )

    (
        matched_strategy,
        detected_alias,
        match_type
    ) = match_strategy(
        extracted_strategy
    )

    # ========================================================
    # 成功匹配
    # ========================================================

    if matched_strategy:

        strategy_info = (
            STRATEGY_LEVEL_MAP[
                matched_strategy
            ]
        )

        strategy_score = (
            strategy_info[
                "score"
            ]
        )

        strategy_level = (
            strategy_info[
                "level"
            ]
        )

        strategy_score_is_default = False

    # ========================================================
    # 无法确定具体策略
    # ========================================================

    else:

        strategy_score = (
            UNMATCHED_STRATEGY_DEFAULT_SCORE
        )

        strategy_level = None

        strategy_score_is_default = True

        # ----------------------------------------------------
        # 只有大类
        # ----------------------------------------------------

        if match_type == "category_only":

            warning_type = (
                "category_only_strategy"
            )

            message = (
                f"只有策略大类，无法确定具体策略 | "
                f"category={detected_alias}"
            )

        # ----------------------------------------------------
        # 没有 Strategy
        # ----------------------------------------------------

        elif match_type == "missing":

            warning_type = (
                "missing_strategy"
            )

            message = (
                "thinking 中没有提取到 Strategy"
            )

        # ----------------------------------------------------
        # 其他无法识别
        # ----------------------------------------------------

        else:

            warning_type = (
                "unmatched_strategy"
            )

            message = (
                f"无法识别具体 Strategy | "
                f"extracted={extracted_strategy}"
            )

        add_warning(
            warning_list=warning_list,

            warning_type=warning_type,

            item=item,

            line_number=line_number,

            message=message,

            extracted_strategy=(
                extracted_strategy
            ),

            detected_alias=(
                detected_alias
            ),

            match_type=(
                match_type
            ),
        )


    rag_score = (
        sample_score
        + emotion_score
        + strategy_score
    ) / 3.0

    # ========================================================
    # 是否需要 RAG
    # ========================================================

    need_rag = (
        1
        if rag_score >= threshold
        else 0
    )

    # ========================================================
    # 新增字段
    # ========================================================

    item[
        "rag_sample_type_score"
    ] = round(
        sample_score,
        4
    )

    item[
        "rag_emotion_score"
    ] = round(
        emotion_score,
        4
    )

    # 原始 Strategy 内容
    item[
        "rag_extracted_strategy"
    ] = extracted_strategy

    # 实际检测到的 ED alias
    item[
        "rag_detected_strategy_alias"
    ] = detected_alias

    # 映射后的标准策略
    item[
        "rag_strategy"
    ] = matched_strategy

    item[
        "rag_strategy_match_type"
    ] = match_type

    item[
        "rag_strategy_level"
    ] = strategy_level

    item[
        "rag_strategy_score"
    ] = round(
        strategy_score,
        4
    )

    # 是否是无法识别后默认给的 0
    item[
        "rag_strategy_score_is_default"
    ] = strategy_score_is_default

    item[
        "rag_score"
    ] = round(
        rag_score,
        4
    )

    item[
        "rag_threshold"
    ] = threshold

    item[
        "need_rag"
    ] = need_rag

    return item


# ============================================================
# 13. 保存 warning 明细
# ============================================================

def save_warning_jsonl(
    warning_list,
    output_path
):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as fout:

        for item in warning_list:

            fout.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# 14. 保存 warning 汇总
# ============================================================

def save_warning_summary(
    warning_list,
    output_path,
    total
):

    type_counter = Counter()

    strategy_counter = Counter()

    category_counter = Counter()

    # ========================================================
    # 统计
    # ========================================================

    for warning in warning_list:

        warning_type = (
            warning.get(
                "warning_type",
                "unknown"
            )
        )

        type_counter[
            warning_type
        ] += 1

        # ----------------------------------------------------
        # Strategy 类 warning
        # ----------------------------------------------------

        if warning_type in {
            "unmatched_strategy",
            "missing_strategy",
            "category_only_strategy",
        }:

            strategy = warning.get(
                "extracted_strategy"
            )

            if strategy is None:

                strategy = "<NONE>"

            strategy_counter[
                strategy
            ] += 1

        # ----------------------------------------------------
        # 只有 category
        # ----------------------------------------------------

        if (
            warning_type
            == "category_only_strategy"
        ):

            category = warning.get(
                "detected_alias"
            )

            category_counter[
                category
            ] += 1

    # ========================================================
    # 写文件
    # ========================================================

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as fout:

        fout.write(
            "=" * 100
            + "\n"
        )

        fout.write(
            "WARNING SUMMARY\n"
        )

        fout.write(
            "=" * 100
            + "\n\n"
        )

        fout.write(
            f"总样本数: {total}\n"
        )

        fout.write(
            f"Warning 总数: "
            f"{len(warning_list)}\n\n"
        )

        # ----------------------------------------------------
        # 类型统计
        # ----------------------------------------------------

        fout.write(
            "【Warning 类型统计】\n"
        )

        fout.write(
            "-" * 100
            + "\n"
        )

        for key, count in (
            type_counter.most_common()
        ):

            fout.write(
                f"{count}\t{key}\n"
            )

        # ----------------------------------------------------
        # Category only
        # ----------------------------------------------------

        fout.write(
            "\n"
        )

        fout.write(
            "【只有大类的 Strategy】\n"
        )

        fout.write(
            "-" * 100
            + "\n"
        )

        for key, count in (
            category_counter.most_common()
        ):

            fout.write(
                f"{count}\t{key}\n"
            )

        # ----------------------------------------------------
        # 剩余 Strategy
        # ----------------------------------------------------

        fout.write(
            "\n"
        )

        fout.write(
            "【仍然没有成功映射的 Strategy】\n"
        )

        fout.write(
            "-" * 100
            + "\n"
        )

        for key, count in (
            strategy_counter.most_common()
        ):

            fout.write(
                f"{count}\t{key}\n"
            )


# ============================================================
# 15. 处理整个 JSONL
# ============================================================

def process_jsonl(
    input_path,
    output_path,
    threshold,
    warning_jsonl_path,
    warning_summary_path
):

    total = 0

    need_rag_count = 0
    no_rag_count = 0

    score_sum = 0.0

    warning_list = []

    # ========================================================
    # Match type 统计
    # ========================================================

    match_type_counter = Counter()

    strategy_counter = Counter()

    # ========================================================
    # 读取
    # ========================================================

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as fin, open(
        output_path,
        "w",
        encoding="utf-8"
    ) as fout:

        for (
            line_number,
            line
        ) in enumerate(
            fin,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            # ------------------------------------------------
            # JSON parse
            # ------------------------------------------------

            try:

                item = json.loads(
                    line
                )

            except json.JSONDecodeError as e:

                warning_list.append(
                    {
                        "warning_type":
                            "json_decode_error",

                        "line_number":
                            line_number,

                        "message":
                            str(e),

                        "raw_line":
                            line,
                    }
                )

                continue

            # ------------------------------------------------
            # Process
            # ------------------------------------------------

            item = process_item(
                item=item,

                threshold=threshold,

                warning_list=warning_list,

                line_number=line_number,
            )

            # ------------------------------------------------
            # 输出
            # ------------------------------------------------

            fout.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )

            total += 1

            score_sum += (
                item[
                    "rag_score"
                ]
            )

            if item[
                "need_rag"
            ] == 1:

                need_rag_count += 1

            else:

                no_rag_count += 1

            # ------------------------------------------------
            # Strategy match 类型统计
            # ------------------------------------------------

            match_type_counter[
                item[
                    "rag_strategy_match_type"
                ]
            ] += 1

            strategy = item.get(
                "rag_strategy"
            )

            if strategy:

                strategy_counter[
                    strategy
                ] += 1

    # ========================================================
    # 保存 Warning
    # ========================================================

    save_warning_jsonl(
        warning_list,
        warning_jsonl_path
    )

    save_warning_summary(
        warning_list,
        warning_summary_path,
        total
    )

    # ========================================================
    # 打印统计
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "处理完成"
    )

    print(
        "=" * 80
    )

    print(
        f"输入文件:"
        f"\n{input_path}"
    )

    print()

    print(
        f"输出文件:"
        f"\n{output_path}"
    )

    print()

    print(
        f"阈值: {threshold}"
    )

    print(
        f"总样本数: {total}"
    )

    print(
        "-" * 80
    )

    if total > 0:

        print(
            f"需要 RAG: "
            f"{need_rag_count} "
            f"("
            f"{need_rag_count / total * 100:.2f}"
            f"%)"
        )

        print(
            f"不需要 RAG: "
            f"{no_rag_count} "
            f"("
            f"{no_rag_count / total * 100:.2f}"
            f"%)"
        )

        print(
            f"平均 rag_score: "
            f"{score_sum / total:.4f}"
        )

    print(
        "-" * 80
    )

    print(
        "Strategy 匹配情况："
    )

    for match_type, count in (
        match_type_counter.most_common()
    ):

        print(
            f"{match_type:25s}: "
            f"{count}"
        )

    print(
        "-" * 80
    )

    print(
        "标准 Strategy 分布："
    )

    for strategy, count in (
        strategy_counter.most_common()
    ):

        score = (
            STRATEGY_LEVEL_MAP[
                strategy
            ]["score"]
        )

        print(
            f"{strategy:35s} "
            f"{count:5d} "
            f"score={score}"
        )

    print(
        "-" * 80
    )

    print(
        f"剩余 Warning 数: "
        f"{len(warning_list)}"
    )

    print()

    print(
        f"Warning 明细:"
        f"\n{warning_jsonl_path}"
    )

    print()

    print(
        f"Warning 汇总:"
        f"\n{warning_summary_path}"
    )

    print(
        "=" * 80
    )


# ============================================================
# 16. main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "根据 sample_type、emotion、strategy "
            "计算 RAG 分数"
        )
    )

    # ========================================================
    # Input
    # ========================================================

    parser.add_argument(
        "--input",

        type=str,

        default=(
            "/user_home/jihongru/jihongru/"
            "empathetic/ACL-Trans/"
            "best_data/ED/ED_bestdata/"
            "7_test_seed43.jsonl"
        ),

        help="输入 JSONL"
    )

    # ========================================================
    # Output
    # ========================================================

    parser.add_argument(
        "--output",

        type=str,

        default=None,

        help="输出 JSONL"
    )

    # ========================================================
    # Threshold
    # ========================================================

    parser.add_argument(
        "--threshold",

        type=float,

        default=0.6,

        choices=[
            0.3,
            0.4,
            0.6,
            0.7,
            0.8,
        ],

        help="RAG threshold"
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    # ========================================================
    # 自动输出文件名
    # ========================================================

    if args.output is None:

        output_path = (
            input_path.with_name(
                f"{input_path.stem}"
                f"_rag_threshold_"
                f"{args.threshold}"
                f"{input_path.suffix}"
            )
        )

    else:

        output_path = Path(
            args.output
        )

    # ========================================================
    # Warning 文件
    # ========================================================

    warning_jsonl_path = (
        output_path.with_name(
            f"{output_path.stem}"
            f"_warnings.jsonl"
        )
    )

    warning_summary_path = (
        output_path.with_name(
            f"{output_path.stem}"
            f"_warning_summary.txt"
        )
    )

    # ========================================================
    # Run
    # ========================================================

    process_jsonl(
        input_path=str(
            input_path
        ),

        output_path=str(
            output_path
        ),

        threshold=(
            args.threshold
        ),

        warning_jsonl_path=str(
            warning_jsonl_path
        ),

        warning_summary_path=str(
            warning_summary_path
        ),
    )


# ============================================================
# 17. Entry
# ============================================================

if __name__ == "__main__":
    main()