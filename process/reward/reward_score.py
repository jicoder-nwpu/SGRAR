# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
# from  import evaluator_bleu_dist_rouge

_SOLUTION_CLIP_CHARS = 300

from transformers import AutoTokenizer
from typing import Optional


def format_reward(predict_str: str) -> float:

    pattern = re.compile(r"<thinking>.*</thinking><answer>.*</answer>$", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    
    score = 1.0 if (match_result and predict_str.count("</answer>") == 1) else 0

    return score


def extract_answer(text):
    answer = text.split("<trend>")[1]
    answer = answer.split("</trend>")[0]
    return answer.strip()


# 策略白名单（严格匹配的标准策略列表）
STRATEGY_WHITELIST = {
    "Exploring Thoughts and Cognition",
    "Exploring Actions and Intentions",
    "Exploring Feelings and Emotions",
    "Restatement or Paraphrasing",
    "Reflection of Feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Positive Reinforcement",
    "Gratitude Prompting",
    "Neutral Validation",
    "Providing Suggestions",
    "Information",
    "Cognitive Reframing",
    "Others"
}

# 标准策略映射（别名/变体 -> 标准策略名）
# 可补充策略的常见别名，比如缩写、大小写变体等
ALIAS_2_STD = {
    "exploring thoughts and cognition": "Exploring Thoughts and Cognition",
    "exploring actions and intentions": "Exploring Actions and Intentions",
    "exploring feelings and emotions": "Exploring Feelings and Emotions",
    "restatement or paraphrasing": "Restatement or Paraphrasing",
    "reflection of feelings": "Reflection of Feelings",
    "self-disclosure": "Self-disclosure",
    "affirmation and reassurance": "Affirmation and Reassurance",
    "positive reinforcement": "Positive Reinforcement",
    "gratitude prompting": "Gratitude Prompting",
    "neutral validation": "Neutral Validation",
    "providing suggestions": "Providing Suggestions",
    "information": "Information",
    "cognitive reframing": "Cognitive Reframing",
    "others": "Others",
}

# 生成匹配用的正则模式（按长度倒序，避免短词匹配覆盖长词）
STD_KEYS = sorted(STRATEGY_WHITELIST, key=len, reverse=True)
pattern = re.compile('|'.join(map(re.escape, STD_KEYS)), flags=re.IGNORECASE)

# ===================== 核心函数 =====================
def extract_sub(text: str) -> str | None:
    """
    提取文本中的标准策略（仅返回白名单内的策略）
    :param text: 待提取的文本
    :return: 匹配到的标准策略名（白名单内），无则返回None
    """
    # 第一步：正则匹配标准策略名
    m = pattern.search(text)
    if m:
        key = m.group(0).strip()
        # 校验是否在白名单内（忽略大小写）
        std_key = next((k for k in STRATEGY_WHITELIST if k.lower() == key.lower()), None)
        if std_key:
            return std_key
    
    # 第二步：兜底匹配别名表
    text_lower = text.lower().strip()
    for alias, std_key in ALIAS_2_STD.items():
        if alias in text_lower:
            # 确保别名映射的结果在白名单内
            if std_key in STRATEGY_WHITELIST:
                return std_key
    
    return None

def extract_answer_strategy(predict_str: str) -> str | None:
    """
    从predict_str中提取answer标签内括号里的策略文本
    匹配格式：</thinking><answer>listener: (策略名称) 内容</answer>
    """
    # 正则匹配第一个answer标签内括号中的内容
    ans_pattern = re.compile(r'</thinking><answer>listener:\s*\(([^)]+)\)', re.IGNORECASE)
    match = ans_pattern.search(predict_str)
    if match:
        # 提取括号内的文本并清理空格
        strategy_text = match.group(1).strip()
        return strategy_text
    return None

def strategy_reward(predict_str: str, strategy: str) -> float:
    """
    计算策略匹配奖励：
    1. 提取answer中括号内的策略文本
    2. 校验是否属于白名单内的标准策略
    3. 与目标策略比对，匹配则返回0.5，否则返回0
    """
    # 步骤1：提取answer中的策略文本
    predict_strategy_text = extract_answer_strategy(predict_str)
    if not predict_strategy_text:
        return 0.0
    
    # 步骤2：提取标准策略（仅白名单内）
    predict_std_strategy = extract_sub(predict_strategy_text)
    print(predict_std_strategy)
    if not predict_std_strategy:
        return 0.0
    
    # 步骤3：标准化目标策略并比对（目标策略需先转为白名单内的标准名）
    target_std_strategy = extract_sub(strategy)
    if not target_std_strategy:
        return 0.0
    
    # 步骤4：返回奖励值
    return 0 if predict_std_strategy != target_std_strategy else 1

EMOTION_PATTERN = re.compile(r"<Emotion>\s*([^\n<]+)", re.IGNORECASE)
def extract_emotion_from_thinking(thinking: str) -> Optional[str]:
    """从thinking字段中提取情绪"""
    if not thinking:
        return None
    match = EMOTION_PATTERN.search(thinking)
    if match:
        return match.group(1).strip()
    return None

def emotion_reward(predict_str: str, emotion: str) -> float:
    predict_emotion = extract_emotion_from_thinking(predict_str)
    if predict_emotion is None:
        return 0
    predict_emotion = predict_emotion.lower()
    emotion = emotion.lower()
    return 1 if (emotion == predict_emotion or emotion in predict_emotion or predict_emotion in emotion) else 0

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    format_score = format_reward(solution_str)

    return format_score
