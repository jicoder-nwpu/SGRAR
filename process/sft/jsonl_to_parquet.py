#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 JSONL 文件转换为 Parquet。

推荐安装：
    pip install pyarrow

示例：
    python jsonl_to_parquet.py \
        --input filtered_part_2693_rag_needed.jsonl \
        --output filtered_part_2693_rag_needed.parquet
"""

import argparse
import os

import pyarrow as pa
import pyarrow.json as pajson
import pyarrow.parquet as pq


def count_jsonl_rows(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="输入 JSONL 文件",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出 Parquet 文件",
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=["snappy", "gzip", "brotli", "zstd", "lz4", "none"],
        help="Parquet 压缩格式，默认 snappy",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        raise ValueError("--input 和 --output 不能相同")

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    jsonl_rows = count_jsonl_rows(args.input)

    if jsonl_rows == 0:
        raise ValueError(f"输入 JSONL 为空: {args.input}")

    print("=" * 72)
    print("JSONL -> Parquet")
    print("=" * 72)
    print(f"input       : {args.input}")
    print(f"output      : {args.output}")
    print(f"compression : {args.compression}")
    print(f"JSONL rows  : {jsonl_rows}")

    table = pajson.read_json(
        args.input,
        read_options=pajson.ReadOptions(
            use_threads=True,
            block_size=64 * 1024 * 1024,
        ),
    )

    compression = None if args.compression == "none" else args.compression

    temp_output = args.output + ".tmp"

    try:
        pq.write_table(
            table,
            temp_output,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )

        check_table = pq.read_table(temp_output)
        parquet_rows = check_table.num_rows

        if parquet_rows != jsonl_rows:
            raise RuntimeError(
                f"行数不一致: JSONL={jsonl_rows}, Parquet={parquet_rows}"
            )

        os.replace(temp_output, args.output)

    except Exception:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        raise

    print(f"Parquet rows: {parquet_rows}")
    print(f"columns     : {table.column_names}")
    print(f"saved       : {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()
