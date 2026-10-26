import pandas as pd
import json
import numpy as np
import argparse

def parquet_to_jsonl(parquet_path, jsonl_path):
    df = pd.read_parquet(parquet_path)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = row.to_dict()

            # ✅ 转换 numpy 类型到原生 Python
            for k, v in record.items():
                if isinstance(v, np.ndarray):
                    record[k] = v.tolist()  # ndarray → list
                elif isinstance(v, (np.integer, np.floating)):
                    record[k] = v.item()    # numpy scalar → python int/float

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入 parquet 文件路径")
    parser.add_argument("--output", required=True, help="输出 jsonl 文件路径")
    args = parser.parse_args()

    parquet_to_jsonl(args.input, args.output)
    print(f"✅ 转换完成！保存到 {args.output}")
