import json


def parse_ground_truth(gt_text: str):
    """
    解析 reward_model["ground_truth"]字符串
    返回: context, emotion, strategy, answer
    """
    # 1. 截取 <thinking> 与 </thinking> 中间部分
    think_start = gt_text.find("<thinking>")
    think_end = gt_text.find("</thinking>")
    think_block = gt_text[think_start + len("<thinking>"): think_end].strip()

    # 3. think_block 按换行切分，过滤空行
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

    return context, emotion, strategy


def process_one_sample(sample: dict):
    # 拼接prompt全部content
    prompt_contents = [item["content"] for item in sample["prompt"]]
    prompt_combined = " \n".join(prompt_contents)

    gt_str = sample["reward_model"]["ground_truth"]
    ctx_text, emo_text, strategy_text = parse_ground_truth(gt_str)

    # extra_info新增字段，context拼接后缀"千百年的"
    sample["extra_info"]["abstract"] = ctx_text
    sample["extra_info"]["hash_id"] = sample['hash_id']
    sample["extra_info"]["sub_conv_id"] = sample['sub_conv_id']
    sample["extra_info"]["turn_id"] = sample['turn_id']
    sample["extra_info"]["data_source"] = sample['data_source']
    sample["extra_info"]["context"] = ctx_text + " " + prompt_combined
    sample["extra_info"]["action"] = strategy_text
    sample["extra_info"]["example"] = "Context: " + ctx_text + " Emotion: " + emo_text + ". Action: " + strategy_text + " Answer: " + sample["extra_info"]["answer"]
    # 可选：把emotion也写入extra_info（原始已有emotion，可覆盖或不处理）
    # sample["extra_info"]["parsed_emotion"] = emo_text

    return sample["extra_info"]


def process_jsonl(input_path: str, output_path: str):
    bad_count = 0
    total = 0
    strategys = set()
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for idx, raw_line in enumerate(fin, start=1):
            total +=1
            line = raw_line.strip()
            # 直接跳过完全空白行
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                bad_count +=1
                print(f"[WARN] line {idx} json decode failed, skip. error: {e}")
                continue

            processed_obj = process_one_sample(obj)
            strategys.add(processed_obj['strategy'])
            fout.write(json.dumps(processed_obj, ensure_ascii=False) + "\n")
    print(strategys)
    print(f"finish. total lines:{total}, bad lines skipped:{bad_count}")


if __name__ == "__main__":
    INPUT = "./ED-database.jsonl"
    OUTPUT = "./ED-database-processed.jsonl"
    process_jsonl(INPUT, OUTPUT)
    print(f"Done. output: {OUTPUT}")
