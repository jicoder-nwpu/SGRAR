import json

def parse_ground_truth(gt_text: str):
    """
    适配ESC数据集：thinking字符串，无外层<thinking></thinking>标签
    输入示例:
    "<Context> The speaker is experiencing mild anger...\n<Emotion> Anger \n<Strategy> Question"
    返回: context, emotion, strategy
    """
    think_block = gt_text.strip()
    lines = [line.strip() for line in think_block.split("\n") if line.strip()]
    context = ""
    emotion = ""
    strategy = ""
    for line in lines:
        if line.startswith("<Context>"):
            context = line[len("<Context>"):].strip()
        elif line.startswith("<Emotion>"):
            emotion = line[len("<Emotion>"):].strip()
        elif line.startswith("<Strategy>"):
            strategy = line[len("<Strategy>"):].strip()
    # 兜底：strategy为空时填充general understanding，对齐检索模板
    if not strategy:
        strategy = "general understanding"
    return context, emotion, strategy


def process_one_sample(sample: dict):
    # ESC数据集key为 "thinking"，不再是reward_model["ground_truth"]
    gt_str = sample["thinking"]
    ctx_text, emo_text, strategy_text = parse_ground_truth(gt_str)

    cur_example = {}
    cur_example["abstract"] = ctx_text
    cur_example["hash_id"] = sample['conv_id']
    cur_example["sub_conv_id"] = sample['sub_conv_id']
    cur_example["turn_id"] = sample['turn_id']
    cur_example["answer"] = sample['output']
    cur_example["action"] = strategy_text
    cur_example["emotion"] = emo_text
    cur_example["strategy"] = strategy_text
    cur_example["context"] = ctx_text + " " + sample['input']
    cur_example["example"] = (
        "\n<Context>\n" + sample['input'] + "\n\n<Summary> " + ctx_text + " \n<Emotion> " + emo_text + "\n<answer>" + sample["output"] + " \n"
    )
    return cur_example


def process_jsonl(input_path: str, output_path: str):
    bad_count = 0
    total = 0
    actionos = set()
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for idx, raw_line in enumerate(fin, start=1):
            total += 1
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                bad_count += 1
                print(f"[WARN] line {idx} json decode failed, skip. error: {e}")
                continue
            processed_obj = process_one_sample(obj)
            actionos.add(processed_obj["strategy"])
            fout.write(json.dumps(processed_obj, ensure_ascii=False) + "\n")
    print("All parsed strategies:")
    print(actionos)
    print(f"finish. total lines:{total}, bad lines skipped:{bad_count}")


if __name__ == "__main__":
    INPUT = "./ESC-database.jsonl"
    OUTPUT = "./ESC-database-processed.jsonl"
    process_jsonl(INPUT, OUTPUT)
    print(f"Done. output: {OUTPUT}")
